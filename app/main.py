import json
import os
from collections.abc import Iterator

import httpx2
from fastapi import Depends, FastAPI, HTTPException
from huggingface_hub import InferenceClient
from huggingface_hub.errors import InferenceTimeoutError
from pydantic import BaseModel, ValidationError

from app.query_parser import InvalidLLMResponse, generate_answer, parse_query
from app.repository import MovieRecord, MovieRepository
from app.schemas import ChatRequest, MovieQuery

app = FastAPI(
    title="Movie Assistant",
    description="A movie recommendation and information API.",
    version="1.0.0",
)
repository = MovieRepository()


class ChatResponse(BaseModel):
    answer: str
    movies: list[MovieRecord]


def get_llm_client() -> Iterator[InferenceClient]:
    # A fixed dummy token prevents the SDK from using any saved Hub credentials.
    with InferenceClient(
        base_url=os.getenv("LLM_BASE_URL", "http://127.0.0.1:11434/v1"),
        token="local",
        timeout=120.0,
    ) as client:
        yield client


@app.post("/movies/search", response_model=list[MovieRecord])
def search_movies(query: MovieQuery) -> list[MovieRecord]:
    return repository.search(query)


@app.post("/movies/chat", response_model=ChatResponse)
def movie_chat(
    request: ChatRequest, client: InferenceClient = Depends(get_llm_client)
) -> ChatResponse:
    try:
        query = parse_query(client, request.message, request.history)
        if query is None:
            return ChatResponse(
                answer="Please ask a movie question using a title, genre, director "
                "or release year. If this is a follow-up, please name the movie "
                "or clarify the filters you want.",
                movies=[],
            )

        movies = repository.search(query)
        if not movies:
            return ChatResponse(
                answer="I couldn't find matching movies in this dataset. "
                "Try a different title or fewer filters.",
                movies=[],
            )

        answer = generate_answer(client, request.message, request.history, movies)
        return ChatResponse(answer=answer, movies=movies)
    except (InferenceTimeoutError, httpx2.TimeoutException) as exc:
        raise HTTPException(status_code=504, detail="Movie assistant timed out") from exc
    except httpx2.ConnectError as exc:
        raise HTTPException(
            status_code=503,
            detail="Local model server is unavailable. Start Ollama with 'ollama serve'.",
        ) from exc
    except (InvalidLLMResponse, ValidationError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=502, detail="Movie assistant returned an invalid response"
        ) from exc
    except httpx2.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail="Local model request failed. Check Ollama and the installed model.",
        ) from exc
