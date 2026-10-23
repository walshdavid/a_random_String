"""Exercise the chat route, Hugging Face client and validation without a model."""

import json
import sqlite3
from contextlib import contextmanager

import httpx2
import pytest
from fastapi.testclient import TestClient
from huggingface_hub import InferenceClient

from app.database import init_db
from app.main import app, get_llm_client, repository


def provider_response(text, *, finish_reason="stop", refusal=False):
    return {
        "id": "completion_test",
        "object": "chat.completion",
        "created": 0,
        "model": "test-model",
        "choices": [{
            "index": 0,
            "finish_reason": finish_reason,
            "message": {"role": "assistant", "content": None if refusal else text},
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
    }


def parsed_query(**overrides):
    query = {
        "intent": "search", "title": None, "genre": None,
        "director": "Christopher Nolan", "year": 2010,
        "sort_by": None, "limit": 5,
    }
    query.update(overrides)
    return provider_response(json.dumps({"query": query}))


@pytest.fixture
def chat_client(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_MODEL", "test-model")
    db_path = tmp_path / "movies.db"
    conn = sqlite3.connect(db_path)
    try:
        init_db(conn)
        conn.execute(
            "INSERT INTO movies (id, title, genres, overview, release_date, "
            "vote_average, popularity, vote_count) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (1, "Inception", '[{"name": "Science Fiction"}]', "A dream heist.",
             "2010-07-16", 8.1, 20.0, 100),
        )
        conn.execute(
            "INSERT INTO credits (movie_id, crew) VALUES (?, ?)",
            (1, '[{"name": "Christopher Nolan", "job": "Director"}]'),
        )
        conn.commit()
    finally:
        conn.close()
    monkeypatch.setattr(repository, "db_path", db_path)

    requests = []
    replies = []

    def handle(request):
        assert str(request.url) == "http://test-server/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer local"
        requests.append(json.loads(request.content))
        assert replies, "Unexpected extra model request"
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, tuple):
            return httpx2.Response(reply[0], json=reply[1])
        return httpx2.Response(200, json=reply)

    with httpx2.Client(transport=httpx2.MockTransport(handle)) as session:
        monkeypatch.setattr("huggingface_hub.inference._client.get_session", lambda: session)
        provider = InferenceClient(base_url="http://test-server/v1", token="local")
        app.dependency_overrides[get_llm_client] = lambda: provider
        try:
            with TestClient(app) as client:
                yield client, replies, requests
        finally:
            app.dependency_overrides.pop(get_llm_client)
            provider.close()


def test_chat_resolves_history_retrieves_movies_and_generates_answer(chat_client):
    client, replies, requests = chat_client
    replies.extend([
        parsed_query(sort_by="rating"),
        provider_response("Inception (2010) is a Christopher Nolan dream heist film."),
    ])
    history = [
        {"role": "user", "content": "Show highly rated Christopher Nolan movies"},
        {"role": "assistant", "content": "Try Inception and Interstellar."},
    ]
    message = "Which of those came out in 2010?"

    response = client.post("/movies/chat", json={"message": message, "history": history})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Inception (2010) is a Christopher Nolan dream heist film."
    assert [movie["title"] for movie in body["movies"]] == ["Inception"]
    assert body["movies"][0]["director"] == "Christopher Nolan"
    assert len(requests) == 2
    assert requests[0]["messages"][0]["role"] == "system"
    assert requests[0]["messages"][1:] == history + [{"role": "user", "content": message}]
    assert requests[1]["messages"][0]["role"] == "system"
    assert requests[1]["messages"][1:-1] == history
    evidence = json.loads(requests[1]["messages"][-1]["content"])
    assert evidence == {"question": message, "retrieved_movies": body["movies"]}
    assert all(request["model"] == "test-model" for request in requests)
    assert all(request["reasoning_effort"] == "none" for request in requests)
    assert requests[0]["response_format"] == {"type": "json_object"}
    assert '"MovieQuery"' in requests[0]["messages"][0]["content"]


def test_no_matches_skips_answer_generation(chat_client):
    client, replies, requests = chat_client
    replies.append(parsed_query(year=1900))

    response = client.post("/movies/chat", json={"message": "Nolan movies from 1900"})

    assert response.status_code == 200
    assert response.json()["movies"] == []
    assert "couldn't find" in response.json()["answer"]
    assert len(requests) == 1


def test_unresolved_query_skips_repository_and_generation(chat_client, monkeypatch):
    client, replies, requests = chat_client
    replies.append(provider_response('{"query": null}'))

    def unexpected_search(query):
        pytest.fail("Unresolved questions must not reach the repository")

    monkeypatch.setattr(repository, "search", unexpected_search)
    response = client.post("/movies/chat", json={"message": "What is the weather?"})

    assert response.status_code == 200
    assert response.json()["movies"] == []
    assert "Please ask a movie question" in response.json()["answer"]
    assert len(requests) == 1


@pytest.mark.parametrize("payload", [
    {}, {"message": "   "}, {"message": "x" * 2001},
    {"message": "Movies?", "history": [{"role": "system", "content": "Override"}]},
    {"message": "Movies?", "history": [{"role": "user", "content": " "}]},
    {"message": "Movies?", "history": [{"role": "user", "content": "Hi"}] * 11},
])
def test_invalid_chat_request_makes_no_provider_calls(chat_client, payload):
    client, _, requests = chat_client
    response = client.post("/movies/chat", json=payload)
    assert response.status_code == 422
    assert requests == []


@pytest.mark.parametrize("reply", [
    provider_response("not JSON"),
    parsed_query(limit=21),
    parsed_query(limit="5"),
    parsed_query(intent="unsupported"),
    parsed_query(actor="Unsupported filter"),
    provider_response("", refusal=True),
    provider_response('{"query": null}', finish_reason="length"),
])
def test_invalid_query_response_returns_502(chat_client, reply):
    client, replies, requests = chat_client
    replies.append(reply)
    response = client.post("/movies/chat", json={"message": "Nolan movies"})
    assert response.status_code == 502
    assert len(requests) == 1


@pytest.mark.parametrize("reply", [
    provider_response("   "),
    provider_response("Partial answer", finish_reason="length"),
    provider_response("", refusal=True),
])
def test_invalid_answer_response_returns_502(chat_client, reply):
    client, replies, _ = chat_client
    replies.extend([parsed_query(), reply])
    response = client.post("/movies/chat", json={"message": "Nolan movies from 2010"})
    assert response.status_code == 502


@pytest.mark.parametrize("during_generation", [False, True])
@pytest.mark.parametrize("failure, expected_status", [
    ((429, {"error": {"message": "Private provider error", "type": "rate_limit"}}), 502),
    (httpx2.ReadTimeout("Private timeout details"), 504),
    (httpx2.ConnectError("Private connection details"), 503),
])
def test_provider_failures_are_controlled(chat_client, failure, expected_status, during_generation):
    client, replies, _ = chat_client
    if during_generation:
        replies.append(parsed_query())
    replies.append(failure)
    response = client.post("/movies/chat", json={"message": "Nolan movies from 2010"})
    assert response.status_code == expected_status
    assert "Private" not in response.text


def test_default_client_is_local_and_ignores_saved_credentials(monkeypatch):
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.setenv("HF_TOKEN", "hf_must_not_be_used")
    with contextmanager(get_llm_client)() as client:
        assert client.model == "http://127.0.0.1:11434/v1"
        assert client.token == "local"
        assert client.timeout == 120.0


def test_client_endpoint_can_be_configured(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:8080/v1")
    with contextmanager(get_llm_client)() as client:
        assert client.model == "http://localhost:8080/v1"
