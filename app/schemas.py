from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints


class MovieQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: Literal["details", "recommend", "search"]
    title: str | None = None
    genre: str | None = None
    director: str | None = None
    year: int | None = None
    sort_by: Literal["rating", "popularity"] | None = None
    limit: int = Field(default=5, ge=1, le=20)


MessageText = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)
]


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: MessageText


class ChatRequest(BaseModel):
    message: MessageText
    history: list[ChatMessage] = Field(default_factory=list, max_length=10)
