import sys

sys.path.append("../")

from typing import Dict, List, Optional

import uuid, tqdm, json, os, random, gc, argparse
from PIL import Image
from google.cloud import bigquery
from pinecone import Pinecone

import src


BATCH_SIZE = 128
NUM_ITEMS = 100000
SHUFFLE_ALPHA = .5
CATALOG_SCORE_ALPHA = .5


def parse_args() -> src.models.CategoryType:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category_type", "-ct", type=str, required=False)
    args = parser.parse_args()

    return args.category_type


def get_gcp_credentials() -> Dict:
    gcp_credentials = secrets.get("GCP_CREDENTIALS")
    gcp_credentials["private_key"] = gcp_credentials["private_key"].replace("\\n", "\n")

    return gcp_credentials


def get_catalog_scores() -> List[int]:
    if random.random() < CATALOG_SCORE_ALPHA:
        return [1, 2, 3]

    return [None]


def get_dataloader(
    category_type: Optional[src.models.CategoryType] = None,
    catalog_score: Optional[int] = None,
    shuffle: bool = False,
) -> bigquery.table.RowIterator:
    return src.bigquery.load_items_to_embed(
        client=bq_client,
        shuffle=shuffle,
        n=NUM_ITEMS,
        category_type=category_type,
        catalog_score=catalog_score,
    )


def upload(points: Dict[str, List[Dict]], rows: Dict[str, List[Dict]]) -> int:
    n_success = 0

    for namespace in points:
        namespace_points = points[namespace]
        namespace_rows = rows[namespace]

        if src.pinecone.upload(
            index=pinecone_index,
            vectors=namespace_points,
            namespace=namespace,
        ):
            if src.bigquery.upload(
                client=bq_client,
                table_id=src.enums.PINECONE_TABLE_ID,
                rows=namespace_rows,
            ):
                n_success = len(namespace_points)

        else:
            valid_rows = []

            for point, row in zip(namespace_points, namespace_rows):
                if src.pinecone.upload(
                    index=pinecone_index,
                    vectors=[point],
                    namespace=namespace,
                ):
                    valid_rows.append(row)

            if src.bigquery.upload(
                client=bq_client,
                table_id=src.enums.PINECONE_TABLE_ID,
                rows=valid_rows,
            ):
                n_success = len(valid_rows)

    return n_success


def main(
    category_type: Optional[src.models.CategoryType] = None,
):
    global secrets
    secrets = json.loads(os.getenv("SECRETS_JSON"))

    global bq_client, pinecone_index
    gcp_credentials = get_gcp_credentials()
    bq_client = src.bigquery.init_client(credentials_dict=gcp_credentials)

    pc_client = Pinecone(api_key=secrets.get("PINECONE_API_KEY"))
    pinecone_index = pc_client.Index(src.enums.PINECONE_INDEX_NAME)
    encoder = src.encoder.FashionCLIPEncoder(normalize=True)

    for catalog_score in get_catalog_scores():
        shuffle = random.random() < SHUFFLE_ALPHA

        print(f"Catalog score: {catalog_score} | Shuffle: {shuffle}")

        n_success, n = 0, 0
        index, point_ids, images, payloads, to_delete_ids = [], [], [], [], []

        loader = get_dataloader(category_type, catalog_score)
        loop = tqdm.tqdm(iterable=loader, total=loader.total_rows)

        for row in loop:
            row = dict(row)
            vinted_id = row.get("vinted_id")

            if vinted_id in index:
                continue

            index.append(vinted_id)
            image_url = row.get("image_location")
            image = src.utils.download_image_as_pil(url=image_url)

            if isinstance(image, Image.Image):
                point_id = str(uuid.uuid4())

                images.append(image)
                payloads.append(row)
                point_ids.append(point_id)

            else:
                to_delete_ids.append(vinted_id)

            if len(point_ids) > 0 and len(point_ids) % BATCH_SIZE == 0:
                n += len(point_ids)

                try:
                    embeddings = encoder.encode_images(images)
                except Exception as e:
                    print(f"Encoding error: {e}")
                    continue

                points, rows = src.pinecone.prepare(
                    point_ids=point_ids, payloads=payloads, embeddings=embeddings
                )

                n_success += upload(points, rows)

                point_ids, images, payloads = [], [], []
                gc.collect()

            success_rate = 0 if n == 0 else n_success / n

            loop.set_description(
                f"Success rate: {success_rate:.2f} | "
                f"Processed: {n} | "
                f"Inserted: {n_success} | "
            )

        if len(to_delete_ids) > 0:
            to_delete_ids = ", ".join([f"'{vinted_id}'" for vinted_id in to_delete_ids])
            conditions = f"vinted_id IN ({to_delete_ids})"

            if src.bigquery.delete(
                client=bq_client,
                table_id=src.enums.ITEM_TABLE_ID,
                conditions=conditions,
            ):
                print(
                    f"Deleted {len(to_delete_ids)} items from {src.enums.ITEM_TABLE_ID}"
                )


if __name__ == "__main__":
    category_type = parse_args()
    main(category_type)
