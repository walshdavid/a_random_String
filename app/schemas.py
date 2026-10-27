from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints


class MovieQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: Literal["details", "recommend", "search"]
    title: str | None = None
    genre: str | None = None
    director: str | None = None
    year: int | None = None
    keywords: list[
        Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]
    ] = Field(default_factory=list, max_length=5)
    sort_by: Literal["rating", "popularity"] | None = None
    limit: int = Field(default=20, ge=1, le=50)


MessageText = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)
]


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: MessageText


class ChatRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "examples": [{
            "message": "Tell me about Inception",
            "conversation_id": "demo",
            "reset_history": False,
        }],
    })

    message: MessageText
    conversation_id: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)
    ] = Field(default="demo", description="ID of the conversation to continue.")
    reset_history: bool = Field(
        default=False, description="Clear this conversation before answering the message."
    )
    history: list[ChatMessage] = Field(
        default_factory=list,
        max_length=10,
        description="Omit to use saved history. If supplied, replaces the context for "
        "this conversation. Ignored when reset_history is true.",
    )
