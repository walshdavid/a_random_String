"""Exercise the chat route and native Ollama requests without a running model."""

import json
import sqlite3
from contextlib import contextmanager

import httpx2
import pytest
from fastapi.testclient import TestClient

from app.model_config import LLMConfig
from app.database import init_db
from app.main import app, get_llm_client, repository


def provider_response(text, *, finish_reason="stop", refusal=False):
    return {
        "model": "test-model",
        "message": {"role": "assistant", "content": None if refusal else text},
        "done": True,
        "done_reason": finish_reason,
    }


def answer_response(text, **kwargs):
    return provider_response(json.dumps({"answer": text}), **kwargs)


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
        assert str(request.url) == "http://test-server/api/chat"
        assert "Authorization" not in request.headers
        requests.append(json.loads(request.content))
        assert replies, "Unexpected extra model request"
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, tuple):
            return httpx2.Response(reply[0], json=reply[1])
        return httpx2.Response(200, json=reply)

    with httpx2.Client(base_url="http://test-server", transport=httpx2.MockTransport(handle)) as provider:
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
        answer_response("Inception (2010) is a Christopher Nolan dream heist film."),
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
    assert all(request["think"] is False for request in requests)
    assert all(request["stream"] is False for request in requests)
    assert "MovieQuery" in requests[0]["format"]["$defs"]
    assert requests[1]["format"]["required"] == ["answer"]
    assert all(request["options"]["temperature"] == 0.7 for request in requests)
    assert all(request["options"]["top_p"] == 0.8 for request in requests)
    assert '"MovieQuery"' in requests[0]["messages"][0]["content"]


def test_no_matches_skips_answer_generation(chat_client):
    client, replies, requests = chat_client
    replies.append(parsed_query(year=1900))

    response = client.post("/movies/chat", json={"message": "Nolan movies from 1900"})

    assert response.status_code == 200
    assert response.json()["movies"] == []
    assert "couldn't find" in response.json()["answer"]
    assert len(requests) == 1


def test_valid_unwrapped_query_is_accepted(chat_client):
    client, replies, _ = chat_client
    # Regression: Qwen returned this shape for "Tell me about inception."
    replies.extend([
        provider_response(json.dumps({
            "intent": "details", "title": "inception", "genre": None,
            "director": None, "year": None, "sort_by": None, "limit": 5,
        })),
        answer_response("Inception is a 2010 film directed by Christopher Nolan."),
    ])
    response = client.post("/movies/chat", json={
        "message": "Tell me about inception.",
        "history": [{"role": "user", "content": "string"}],
    })
    assert response.status_code == 200
    assert response.json()["movies"][0]["title"] == "Inception"
    assert response.json()["answer"] == "Inception is a 2010 film directed by Christopher Nolan."


@pytest.mark.parametrize("prefix", ["<think>Internal text</think>", "Internal text</think>"])
def test_thinking_text_is_not_returned(chat_client, prefix):
    client, replies, requests = chat_client
    replies.extend([parsed_query(), answer_response(prefix + "\nChristopher Nolan directed it.")])
    response = client.post("/movies/chat", json={"message": "Who directed Inception?"})
    assert response.status_code == 200
    assert response.json()["answer"] == "Christopher Nolan directed it."
    assert all(request["think"] is False for request in requests)


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
    provider_response('{"intent": "search", "limit": 21}'),
    provider_response('{"intent": "search", "actor": "Unsupported"}'),
    provider_response('{"intent": "search", "year": "2010"}'),
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
    answer_response("   "),
    answer_response("Partial answer", finish_reason="length"),
    provider_response("", refusal=True),
    answer_response("<think>Unfinished internal text"),
    answer_response("<think>Internal text</think>"),
    provider_response("not JSON"),
    provider_response('{"answer": 123}'),
])
def test_invalid_answer_response_returns_502(chat_client, reply):
    client, replies, _ = chat_client
    replies.extend([parsed_query(), reply])
    response = client.post("/movies/chat", json={"message": "Nolan movies from 2010"})
    assert response.status_code == 502


@pytest.mark.parametrize("during_generation", [False, True])
@pytest.mark.parametrize("failure, expected_status", [
    ((429, {"error": {"message": "Private provider error", "type": "rate_limit"}}), 502),
    ((500, {"error": "prediction aborted, token repeat limit reached"}), 502),
    (httpx2.ReadTimeout("Private timeout details"), 504),
    (httpx2.ConnectError("Private connection details"), 503),
])
def test_provider_failures_are_controlled(chat_client, caplog, failure, expected_status, during_generation):
    client, replies, _ = chat_client
    if during_generation:
        replies.append(parsed_query())
    replies.append(failure)
    response = client.post("/movies/chat", json={"message": "Nolan movies from 2010"})
    assert response.status_code == expected_status
    assert "Private" not in response.text
    assert len(chat_client[2]) == (2 if during_generation else 1)
    if isinstance(failure, tuple):
        assert ("answer generation" if during_generation else "query parsing") in caplog.text


def test_default_client_is_local_and_ignores_saved_credentials(monkeypatch):
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.setenv("HF_TOKEN", "hf_must_not_be_used")
    with contextmanager(get_llm_client)(LLMConfig.from_env()) as client:
        assert str(client.base_url) == "http://127.0.0.1:11434"
        assert "Authorization" not in client.headers
        assert client.timeout.read == 120.0


def test_client_endpoint_can_be_configured(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:8080/v1")
    with contextmanager(get_llm_client)(LLMConfig.from_env()) as client:
        assert str(client.base_url) == "http://localhost:8080"


def test_follow_up_uses_request_history(chat_client):
    client, replies, requests = chat_client
    answer = "Inception is a dream heist film directed by Christopher Nolan."
    replies.extend([parsed_query(), answer_response(answer)])
    first = client.post("/movies/chat", json={"message": "Tell me about Inception"})
    assert first.status_code == 200

    replies.extend([parsed_query(), answer_response("It was released in 2010.")])
    follow_up = client.post("/movies/chat", json={
        "message": "When was it released?",
        "history": [
            {"role": "user", "content": "Tell me about Inception"},
            {"role": "assistant", "content": first.json()["answer"]},
        ],
    })

    assert follow_up.status_code == 200
    assert set(follow_up.json()) == {"answer", "movies"}
    assert requests[2]["messages"][1:-1] == [
        {"role": "user", "content": "Tell me about Inception"},
        {"role": "assistant", "content": answer},
    ]


def test_requests_without_history_are_independent(chat_client):
    client, replies, requests = chat_client
    for message in ["Nolan movies", "War movies"]:
        replies.extend([parsed_query(), answer_response("Inception.")])
        response = client.post("/movies/chat", json={"message": message})
        assert response.status_code == 200

    assert requests[2]["messages"][1:] == [{"role": "user", "content": "War movies"}]


def test_ten_history_messages_are_accepted(chat_client):
    client, replies, requests = chat_client
    replies.append(provider_response('{"query": null}'))
    history = [
        {"role": "user" if index % 2 == 0 else "assistant", "content": f"Message {index}"}
        for index in range(10)
    ]
    response = client.post("/movies/chat", json={
        "message": "Tell me more", "history": history,
    })
    assert response.status_code == 200
    assert len(requests) == 1
    assert requests[0]["messages"][1:-1] == history
