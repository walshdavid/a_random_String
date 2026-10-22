from typing import Literal
from pydantic import BaseModel, Field


class MovieQuery(BaseModel):
    intent: Literal["details", "recommend", "search"]
    title: str | None = None
    genre: str | None = None
    director: str | None = None
    year: int | None = None
    sort_by: Literal["rating", "popularity"] | None = None
    limit: int = Field(default=5, ge=1, le=20)