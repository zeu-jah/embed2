from typing import List, Literal
from pydantic import BaseModel, Field


CategoryType = Literal[
    "top",
    "accessories",
    "bottom",
    "outerwear",
    "footwear",
    "dress",
    "suit"
]


class ColorVector(BaseModel):
    id: int
    title: str
    values: List[float] = Field(..., min_items=1)

    @classmethod
    def from_dict(cls, data: dict) -> "ColorVector":
        return cls(
            id=data["id"],
            title=data["title"],
            values=data["values"],
        )