import json
import logging
from collections.abc import Iterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from threading import Lock
from time import gmtime

import httpx2
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
        timeout=30.0,
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
            query = parse_query(client, request.message, history, config)
            logger.info(
                "First LLM call finished; MovieQuery: %s",
                query.model_dump_json() if query is not None else None,
            )
            movies = []
            if query is not None:
                stage = "database search"
                logger.info("Database search started")
                movies = repository.search(query)
                logger.info(
                    "Database search finished; records: %s",
                    json.dumps([asdict(movie) for movie in movies]),
                )
            if query is None:
                answer = (
                    "Please ask a movie question using a title, genre, director "
                    "or release year. If this is a follow-up, please name the movie "
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
                answer = generate_answer(client, request.message, history, movies, config)
                logger.info("Second LLM call finished")

            response = ChatResponse(
                conversation_id=request.conversation_id, answer=answer, movies=movies
            )
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
            "%(asctime)s.%(msecs)03dZ %(levelname)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
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
