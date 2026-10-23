import json
import logging
from collections.abc import Iterator
from contextlib import asynccontextmanager

import httpx2
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from huggingface_hub import InferenceClient
from huggingface_hub.errors import InferenceTimeoutError
from pydantic import BaseModel, ValidationError

from app.model_config import LLMConfig
from app.query_parser import InvalidLLMResponse, generate_answer, parse_query
from app.repository import MovieRecord, MovieRepository
from app.schemas import ChatRequest, MovieQuery

router = APIRouter()
repository = MovieRepository()
logger = logging.getLogger(__name__)


class ChatResponse(BaseModel):
    answer: str
    movies: list[MovieRecord]


def get_llm_config(request: Request) -> LLMConfig:
    return request.app.state.llm_config


def get_llm_client(
    config: LLMConfig = Depends(get_llm_config),
) -> Iterator[InferenceClient | httpx2.Client]:
    if config.mode == "local":
        # Accept the old /v1 setting, but use Ollama's native /api/chat endpoint.
        assert config.base_url is not None
        base_url = config.base_url.rstrip("/").removesuffix("/v1")
        with httpx2.Client(base_url=base_url, timeout=120.0, trust_env=False) as local_client:
            yield local_client
        return

    with InferenceClient(
        base_url=None,
        provider="auto",
        token=config.token,
        timeout=30.0,
    ) as client:
        yield client


@router.post("/movies/search", response_model=list[MovieRecord])
def search_movies(query: MovieQuery) -> list[MovieRecord]:
    return repository.search(query)


@router.post("/movies/chat", response_model=ChatResponse)
def movie_chat(
    request: ChatRequest,
    client: InferenceClient | httpx2.Client = Depends(get_llm_client),
    config: LLMConfig = Depends(get_llm_config),
) -> ChatResponse:
    stage = "query parsing"
    try:
        query = parse_query(client, request.message, request.history, config)
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

        stage = "answer generation"
        answer = generate_answer(client, request.message, request.history, movies, config)
        return ChatResponse(answer=answer, movies=movies)
    except (InferenceTimeoutError, httpx2.TimeoutException) as exc:
        raise HTTPException(status_code=504, detail="Movie assistant timed out") from exc
    except httpx2.ConnectError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Local model server is unavailable. Start Ollama with 'ollama serve'."
                if config.mode == "local" else "Hosted model server is unavailable."
            ),
        ) from exc
    except (InvalidLLMResponse, ValidationError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=502, detail="Movie assistant returned an invalid response"
        ) from exc
    except httpx2.HTTPError as exc:
        status = exc.response.status_code if isinstance(exc, httpx2.HTTPStatusError) else None
        logger.warning("LLM %s failed: mode=%s, upstream_status=%s", stage, config.mode, status)
        raise HTTPException(
            status_code=502,
            detail=(
                "Local model request failed. Check Ollama and the installed model."
                if config.mode == "local"
                else "Hosted model request failed. Check HF_TOKEN, HF_MODEL and credits."
            ),
        ) from exc


def create_app(config: LLMConfig | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.llm_config = config or LLMConfig.from_env()
        yield

    application = FastAPI(
        title="Movie Assistant",
        description="A movie recommendation and information API.",
        version="1.0.0",
        lifespan=lifespan,
    )
    application.include_router(router)
    return application


app = create_app()
