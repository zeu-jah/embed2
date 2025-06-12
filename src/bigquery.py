from typing import List, Dict, Optional

from google.oauth2 import service_account
from google.cloud import bigquery

from .models import CategoryType
from .enums import *


BASE_QUERY = f"""
SELECT 
item.*, 
FROM `{PROJECT_ID}.{DATASET_ID}.{ITEM_ACTIVE_TABLE_ID}` item
LEFT JOIN `{PROJECT_ID}.{DATASET_ID}.{PINECONE_TABLE_ID}` AS p ON item.id = p.item_id
LEFT JOIN `{PROJECT_ID}.{DATASET_ID}.{SOLD_TABLE_ID}` AS s USING (vinted_id)
WHERE p.item_id IS NULL AND s.vinted_id IS NULL
"""


def init_client(credentials_dict: Dict) -> bigquery.Client:
    credentials = service_account.Credentials.from_service_account_info(
        credentials_dict
    )

    return bigquery.Client(
        credentials=credentials, project=credentials_dict["project_id"]
    )


def load_items_to_embed(
    client: bigquery.Client,
    shuffle: bool = False,
    n: Optional[int] = None,
    category_type: Optional[CategoryType] = None,
) -> bigquery.table.RowIterator:
    query = _query_items_to_embed(
        shuffle=shuffle,
        n=n,
        category_type=category_type,
    )

    return client.query(query).result()


def upload(client: bigquery.Client, table_id: str, rows: List[Dict]) -> bool:
    try:
        if len(rows) == 0:
            return False

        errors = client.insert_rows_json(
            table=f"{PROJECT_ID}.{DATASET_ID}.{table_id}", json_rows=rows
        )

        if not errors:
            return True
        else:
            print(errors)
            return False

    except Exception as e:
        print(e)
        return False


def delete(client: bigquery.Client, table_id: str, conditions: List[str]) -> bool:
    query = f"""
    DELETE FROM `{PROJECT_ID}.{DATASET_ID}.{table_id}`
    WHERE {conditions}
    """

    try:
        client.query(query).result()
        return True
    except Exception as e:
        print(e)
        return False
    

def _query_items_to_embed(
    shuffle: bool = False,
    n: Optional[int] = None,
    category_type: Optional[CategoryType] = None,
) -> str:
    base_query = _build_base_query(category_type)

    if category_type is not None:
        return _build_single_category_query(base_query, shuffle, n)
    else:
        return _build_weighted_category_query(base_query, shuffle, n)


def _build_base_query(category_type: Optional[CategoryType] = None) -> str:
    query = BASE_QUERY

    if category_type is not None:
        query += f" AND category_type = '{category_type}'"
    else:
        category_types = ", ".join(
            [f"'{category_type}'" for category_type in CATEGORY_TYPES]
        )
        query += f" AND category_type IN ({category_types})"

    return query


def _build_single_category_query(
    base_query: str, shuffle: bool, n: Optional[int] = None
) -> str:
    query = base_query

    if shuffle:
        query += "\nORDER BY RAND()"
    else:
        query += "\nORDER BY created_at DESC"

    if n is not None:
        query += f"\nLIMIT {n}"

    return query


def _calculate_category_limits(n: int) -> tuple[int, int]:
    total_weight = (len(MAIN_CATEGORIES) * MAIN_CATEGORY_WEIGHT) + (
        len(CATEGORY_TYPES) - len(MAIN_CATEGORIES)
    ) * OTHER_CATEGORY_WEIGHT
    items_per_weight = n // total_weight

    main_category_limit = items_per_weight * MAIN_CATEGORY_WEIGHT
    other_category_limit = items_per_weight * OTHER_CATEGORY_WEIGHT

    return main_category_limit, other_category_limit


def _build_weighted_category_query(
    base_query: str, shuffle: bool, n: Optional[int] = None
) -> str:
    main_categories_str = ", ".join(f"'{cat}'" for cat in MAIN_CATEGORIES)

    numbered_items_query = f"""
    SELECT 
    *,
    ROW_NUMBER() OVER (
        PARTITION BY category_type ORDER BY {'RAND()' if shuffle else 'created_at DESC'}
    ) as row_num,
    CASE 
    WHEN category_type IN ({main_categories_str}) THEN {MAIN_CATEGORY_WEIGHT}
    ELSE {OTHER_CATEGORY_WEIGHT}
    END as category_weight
    FROM base_items
    """
    
    query = f"""
    WITH 
        base_items AS ({base_query}),
        numbered_items AS ({numbered_items_query})
    SELECT * EXCEPT(row_num, category_weight)
    FROM numbered_items
    """

    if n is not None:
        main_limit, other_limit = _calculate_category_limits(n)
        query += f"""
        WHERE (category_type IN ({main_categories_str}) AND row_num <= {main_limit})
           OR (category_type NOT IN ({main_categories_str}) AND row_num <= {other_limit})
        """

    return query
