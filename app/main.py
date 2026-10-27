"""Start with `uv run python -m app.main` or add `--API` for hosted inference."""

import argparse
import json
import logging
import platform
from collections.abc import Iterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from threading import Lock
from time import gmtime, perf_counter

import httpx2
import uvicorn
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from huggingface_hub import InferenceClient
from huggingface_hub.errors import InferenceTimeoutError
from pydantic import BaseModel, ValidationError

from app.model_config import LLMConfig
from app.query_parser import InvalidLLMResponse, generate_answer, parse_query
from app.repository import MovieRecord, MovieRepository
from app.schemas import ChatMessage, ChatRequest, MovieQuery

router = APIRouter()
repository = MovieRepository()
logger = logging.getLogger(__name__)


class ChatResponse(BaseModel):
    conversation_id: str
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
        timeout=120.0,
    ) as client:
        yield client


@router.post("/movies/search", response_model=list[MovieRecord])
def search_movies(query: MovieQuery) -> list[MovieRecord]:
    return repository.search(query)


@router.post("/movies/chat", response_model=ChatResponse)
def movie_chat(
    request: ChatRequest,
    http_request: Request,
    client: InferenceClient | httpx2.Client = Depends(get_llm_client),
    config: LLMConfig = Depends(get_llm_config),
) -> ChatResponse:
    state = http_request.app.state
    # Serialize chat turns so overlapping requests cannot overwrite saved history.
    with state.conversation_lock:
        if request.reset_history:
            state.conversations.pop(request.conversation_id, None)
            history = []
        elif "history" in request.model_fields_set:
            history = request.history
        else:
            history = state.conversations.get(request.conversation_id, [])
        logger.info("Question: %s", json.dumps(request.message))
        logger.info("History: %s", json.dumps([item.model_dump() for item in history]))

        stage = "query parsing"
        try:
            logger.info("First LLM call started. Request: %s", json.dumps(request.model_dump()))
            started = perf_counter()
            query = parse_query(client, request.message, history, config)
            duration_ms = (perf_counter() - started) * 1000
            logger.info(
                "First LLM call finished; MovieQuery: %s",
                query.model_dump_json() if query is not None else None,
                extra={"duration_ms": f"{duration_ms:.1f}"},
            )
            movies = []
            if query is not None:
                stage = "database search"
                logger.info("Database search started")
                started = perf_counter()
                movies = repository.search(query)
                duration_ms = (perf_counter() - started) * 1000
                logger.info(
                    "Database search finished; records: %s",
                    json.dumps([asdict(movie) for movie in movies]),
                    extra={"duration_ms": f"{duration_ms:.1f}"},
                )
            if query is None:
                answer = (
                    "Please ask a movie question using a title, genre, director "
                    "or release year, or describe a topic using keywords. "
                    "If this is a follow-up, please name the movie "
                    "or clarify the filters you want."
                )
            elif not movies:
                answer = (
                    "I couldn't find matching movies in this dataset. "
                    "Try a different title or fewer filters."
                )
            else:
                stage = "answer generation"
                logger.info("Second LLM call started")
                started = perf_counter()
                answer = generate_answer(client, request.message, history, movies, config)
                logger.info(
                    "Second LLM call finished",
                    extra={"duration_ms": f"{(perf_counter() - started) * 1000:.1f}"},
                )

            response = ChatResponse(
                conversation_id=request.conversation_id, answer=answer, movies=movies
            )
        except (InferenceTimeoutError, httpx2.TimeoutException) as exc:
            logger.exception(
                "LLM %s timed out: mode=%s, model=%s", stage, config.mode, config.model,
                extra={"duration_ms": f"{(perf_counter() - started) * 1000:.1f}"},
            )
            raise HTTPException(status_code=504, detail="Movie assistant timed out") from exc
        except httpx2.ConnectError as exc:
            logger.exception(
                "LLM %s connection failed: mode=%s, model=%s", stage, config.mode, config.model,
                extra={"duration_ms": f"{(perf_counter() - started) * 1000:.1f}"},
            )
            raise HTTPException(
                status_code=503,
                detail=(
                    "Local model server is unavailable. Start Ollama with 'ollama serve'."
                    if config.mode == "local" else "Hosted model server is unavailable."
                ),
            ) from exc
        except (InvalidLLMResponse, ValidationError, json.JSONDecodeError) as exc:
            logger.exception(
                "LLM %s returned an invalid response: mode=%s, model=%s",
                stage, config.mode, config.model,
                extra={"duration_ms": f"{(perf_counter() - started) * 1000:.1f}"},
            )
            raise HTTPException(
                status_code=502, detail="Movie assistant returned an invalid response"
            ) from exc
        except httpx2.HTTPError as exc:
            upstream_response = getattr(exc, "response", None)
            status = upstream_response.status_code if upstream_response is not None else None
            body = upstream_response.text if upstream_response is not None else None
            logger.exception(
                "LLM %s failed: mode=%s, model=%s, upstream_status=%s, response=%s",
                stage, config.mode, config.model, status, body,
                extra={"duration_ms": f"{(perf_counter() - started) * 1000:.1f}"},
            )
            raise HTTPException(
                status_code=502,
                detail=(
                    "Local model request failed. Check Ollama and the installed model."
                    if config.mode == "local"
                    else "Hosted model request failed. Check HF_TOKEN, HF_MODEL and credits."
                ),
            ) from exc

        # Store completed turns only. Bound saved model text like incoming messages;
        # the response itself still contains the full answer.
        state.conversations[request.conversation_id] = (history + [
            ChatMessage(role="user", content=request.message),
            ChatMessage(role="assistant", content=answer[:2000]),
        ])[-10:]
        logger.info("Answer: %s", json.dumps(answer))
        return response


def create_app(config: LLMConfig | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.llm_config = config or LLMConfig.from_env()
        application.state.conversations = {}
        application.state.conversation_lock = Lock()
        handler = logging.FileHandler("log.txt", mode="w", encoding="utf-8")
        formatter = logging.Formatter(
            "%(asctime)s.%(msecs)03dZ %(levelname)s duration_ms=%(duration_ms)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
            defaults={"duration_ms": "-"},
        )
        formatter.converter = gmtime
        handler.setFormatter(formatter)
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
        try:
            yield
        finally:
            logger.removeHandler(handler)
            handler.close()

    application = FastAPI(
        title="Movie Assistant",
        description="A movie recommendation and information API.",
        version="1.0.0",
        lifespan=lifespan,
    )
    application.include_router(router)
    return application


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the movie assistant")
    parser.add_argument("--API", action="store_true", help="Use Hugging Face hosted inference")
    args = parser.parse_args()
    try:
        config = LLMConfig.from_env(use_api=args.API)
    except ValueError as exc:
        parser.error(str(exc))

    print(f"Starting in {config.mode} mode using {config.model}")
    print(f"Python version: {platform.python_version()}")
    uvicorn.run(create_app(config), host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
