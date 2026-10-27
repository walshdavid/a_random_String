"""Exercise the chat route and native Ollama requests without a running model."""

import json
import sqlite3
from contextlib import contextmanager

import httpx2
import pytest
from fastapi.testclient import TestClient

from app.model_config import LLMConfig
from app.database import init_db
from app.main import create_app, get_llm_client, repository


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
        "keywords": [], "sort_by": None, "limit": 5,
    }
    query.update(overrides)
    return provider_response(json.dumps({"query": query}))


@pytest.fixture
def chat_client(monkeypatch, tmp_path):
    monkeypatch.setenv("OLLAMA_MODEL", "test-model")
    app = create_app()
    db_path = tmp_path / "movies.db"
    conn = sqlite3.connect(db_path)
    try:
        init_db(conn)
        conn.execute(
            'INSERT INTO movies (id, title, year, overview, "cast", director, popularity) '
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (1, "Inception", 2010, "A dream heist.", '["Leonardo DiCaprio", "Joseph Gordon-Levitt"]',
             "Christopher Nolan", 20.0),
        )
        conn.execute(
            "INSERT INTO genres (movie_id, name) VALUES (?, ?)",
            (1, "Science Fiction"),
        )
        conn.execute(
            "INSERT INTO ratings (movie_id, vote_average, vote_count) VALUES (?, ?, ?)",
            (1, 8.1, 100),
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


def test_christmas_keyword_retrieves_movies_and_passes_evidence_to_llm2(chat_client):
    client, replies, requests = chat_client
    conn = sqlite3.connect(repository.db_path)
    try:
        conn.execute(
            "INSERT INTO movies (id, title, overview, keywords, popularity) VALUES (?, ?, ?, ?, ?)",
            (2, "Winter Reunion", "Friends reunite for an annual celebration.",
             '["christmas"]', 5),
        )
        conn.commit()
    finally:
        conn.close()

    replies.extend([
        parsed_query(intent="recommend", director=None, year=None, keywords=["christmas"]),
        answer_response("Try Winter Reunion, which is tagged with Christmas."),
    ])
    response = client.post("/movies/chat", json={"message": "Find me a Christmas movie"})

    assert response.status_code == 200
    movies = response.json()["movies"]
    assert [movie["title"] for movie in movies] == ["Winter Reunion"]
    assert movies[0]["keywords"] == ["christmas"]
    assert json.loads(requests[1]["messages"][-1]["content"])["retrieved_movies"] == movies
    keyword_schema = requests[0]["format"]["$defs"]["MovieQuery"]["properties"]["keywords"]
    assert keyword_schema["type"] == "array"


def test_cast_names_are_returned_and_supplied_as_answer_evidence(chat_client):
    client, replies, requests = chat_client
    replies.extend([
        parsed_query(intent="details", title="Inception", director=None, year=None),
        answer_response("Inception stars Leonardo DiCaprio and Joseph Gordon-Levitt."),
    ])

    response = client.post("/movies/chat", json={"message": "Who stars in Inception?"})

    assert response.status_code == 200
    expected_cast = ["Leonardo DiCaprio", "Joseph Gordon-Levitt"]
    assert response.json()["movies"][0]["cast"] == expected_cast
    evidence = json.loads(requests[1]["messages"][-1]["content"])
    assert evidence["retrieved_movies"][0]["cast"] == expected_cast


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
    {"message": "Movies?", "conversation_id": "   "},
    {"message": "Movies?", "conversation_id": "x" * 101},
    {"message": "Movies?", "reset_history": "invalid"},
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
    provider_response('{"intent": "search", "limit": 51}'),
    provider_response('{"intent": "search", "actor": "Unsupported"}'),
    provider_response('{"intent": "search", "year": "2010"}'),
    parsed_query(limit=51),
    parsed_query(limit="5"),
    parsed_query(intent="unsupported"),
    parsed_query(actor="Unsupported filter"),
    parsed_query(keywords="christmas"),
    parsed_query(keywords=[" "]),
    parsed_query(keywords=["x" * 101]),
    parsed_query(keywords=["christmas"] * 6),
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
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.setenv("HF_TOKEN", "hf_must_not_be_used")
    with contextmanager(get_llm_client)(LLMConfig.from_env()) as client:
        assert str(client.base_url) == "http://127.0.0.1:11434"
        assert "Authorization" not in client.headers
        assert client.timeout.read == 120.0


def test_client_endpoint_can_be_configured(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:8080/v1")
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
    assert set(follow_up.json()) == {"conversation_id", "answer", "movies"}
    assert requests[2]["messages"][1:-1] == [
        {"role": "user", "content": "Tell me about Inception"},
        {"role": "assistant", "content": answer},
    ]


def test_follow_up_uses_saved_default_conversation(chat_client):
    client, replies, requests = chat_client
    answer = "Inception is a dream heist film."
    replies.extend([parsed_query(), answer_response(answer)])
    first = client.post("/movies/chat", json={"message": "Tell me about Inception"})
    assert first.status_code == 200
    assert first.json()["conversation_id"] == "demo"

    replies.extend([parsed_query(), answer_response("Christopher Nolan directed it.")])
    follow_up = client.post("/movies/chat", json={"message": "Who directed it?"})

    assert follow_up.status_code == 200
    expected_history = [
        {"role": "user", "content": "Tell me about Inception"},
        {"role": "assistant", "content": answer},
    ]
    assert requests[0]["messages"][1:-1] == []
    assert requests[2]["messages"][1:-1] == expected_history
    assert requests[3]["messages"][1:-1] == expected_history


def test_conversation_ids_keep_histories_separate(chat_client):
    client, replies, requests = chat_client
    for conversation_id in ["alice", "bob"]:
        replies.extend([parsed_query(), answer_response(f"Answer for {conversation_id}.")])
        response = client.post("/movies/chat", json={
            "message": f"Question from {conversation_id}",
            "conversation_id": conversation_id,
        })
        assert response.status_code == 200
        assert response.json()["conversation_id"] == conversation_id
        assert requests[-2]["messages"][1:-1] == []

    replies.extend([parsed_query(), answer_response("A follow-up answer.")])
    response = client.post("/movies/chat", json={
        "message": "Who directed it?", "conversation_id": "alice",
    })
    assert response.status_code == 200
    assert requests[-2]["messages"][1:-1] == [
        {"role": "user", "content": "Question from alice"},
        {"role": "assistant", "content": "Answer for alice."},
    ]


def test_reset_clears_only_selected_conversation(chat_client):
    client, replies, requests = chat_client
    for conversation_id in ["demo", "other"]:
        replies.extend([parsed_query(), answer_response(f"Old {conversation_id} answer.")])
        assert client.post("/movies/chat", json={
            "message": f"Old {conversation_id} question",
            "conversation_id": conversation_id,
        }).status_code == 200

    replies.extend([parsed_query(), answer_response("A new answer.")])
    response = client.post("/movies/chat", json={
        "message": "A new question", "reset_history": True,
        "history": [{"role": "user", "content": "This must also be ignored"}],
    })
    assert response.status_code == 200
    assert requests[-2]["messages"][1:-1] == []
    assert requests[-1]["messages"][1:-1] == []

    for conversation_id, expected_history in [
        ("demo", [
            {"role": "user", "content": "A new question"},
            {"role": "assistant", "content": "A new answer."},
        ]),
        ("other", [
            {"role": "user", "content": "Old other question"},
            {"role": "assistant", "content": "Old other answer."},
        ]),
    ]:
        replies.extend([parsed_query(), answer_response("A follow-up answer.")])
        response = client.post("/movies/chat", json={
            "message": "Follow up", "conversation_id": conversation_id,
        })
        assert response.status_code == 200
        assert requests[-2]["messages"][1:-1] == expected_history


def test_saved_history_keeps_latest_ten_messages(chat_client):
    client, replies, requests = chat_client
    for index in range(7):
        replies.extend([parsed_query(), answer_response(f"Answer {index}.")])
        response = client.post("/movies/chat", json={"message": f"Question {index}"})
        assert response.status_code == 200

    # The seventh turn sees the five preceding exchanges; the oldest is gone.
    expected_history = [
        message
        for index in range(1, 6)
        for message in [
            {"role": "user", "content": f"Question {index}"},
            {"role": "assistant", "content": f"Answer {index}."},
        ]
    ]
    assert requests[-2]["messages"][1:-1] == expected_history
    assert requests[-1]["messages"][1:-1] == expected_history


@pytest.mark.parametrize("during_generation", [False, True])
def test_failed_turn_is_not_saved(chat_client, during_generation):
    client, replies, requests = chat_client
    replies.extend([parsed_query(), answer_response("First answer.")])
    assert client.post("/movies/chat", json={"message": "First question"}).status_code == 200

    if during_generation:
        replies.append(parsed_query())
    replies.append((500, {"error": "Model failed"}))
    assert client.post("/movies/chat", json={"message": "Failed question"}).status_code == 502

    replies.extend([parsed_query(), answer_response("Follow-up answer.")])
    response = client.post("/movies/chat", json={"message": "Follow up"})
    assert response.status_code == 200
    assert requests[-2]["messages"][1:-1] == [
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "First answer."},
    ]


def test_reset_still_clears_history_when_model_fails(chat_client):
    client, replies, requests = chat_client
    replies.extend([parsed_query(), answer_response("Old answer.")])
    assert client.post("/movies/chat", json={"message": "Old question"}).status_code == 200

    replies.append((500, {"error": "Model failed"}))
    assert client.post("/movies/chat", json={
        "message": "New question", "reset_history": True,
    }).status_code == 502

    replies.extend([parsed_query(), answer_response("New answer.")])
    assert client.post("/movies/chat", json={"message": "Retry"}).status_code == 200
    assert requests[-2]["messages"][1:-1] == []


@pytest.mark.parametrize("query_reply", [provider_response('{"query": null}'), parsed_query(year=1900)])
def test_fixed_replies_are_saved_in_history(chat_client, query_reply):
    client, replies, requests = chat_client
    replies.append(query_reply)
    first = client.post("/movies/chat", json={"message": "First question"})
    assert first.status_code == 200
    assert first.json()["movies"] == []

    replies.extend([parsed_query(), answer_response("Follow-up answer.")])
    assert client.post("/movies/chat", json={"message": "Follow up"}).status_code == 200
    assert requests[-2]["messages"][1:-1] == [
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": first.json()["answer"]},
    ]


@pytest.mark.parametrize("history", [[], [{"role": "user", "content": "Replacement context"}]])
def test_explicit_history_replaces_saved_context(chat_client, history):
    client, replies, requests = chat_client
    replies.extend([parsed_query(), answer_response("Old answer.")])
    assert client.post("/movies/chat", json={"message": "Old question"}).status_code == 200

    replies.extend([parsed_query(), answer_response("New answer.")])
    response = client.post("/movies/chat", json={"message": "New question", "history": history})
    assert response.status_code == 200
    assert requests[-2]["messages"][1:-1] == history

    replies.extend([parsed_query(), answer_response("Follow-up answer.")])
    assert client.post("/movies/chat", json={"message": "Follow up"}).status_code == 200
    assert requests[-2]["messages"][1:-1] == history + [
        {"role": "user", "content": "New question"},
        {"role": "assistant", "content": "New answer."},
    ]


def test_long_answer_is_bounded_in_history_but_returned_in_full(chat_client):
    client, replies, requests = chat_client
    answer = "A" * 2100
    replies.extend([parsed_query(), answer_response(answer)])
    first = client.post("/movies/chat", json={"message": "Tell me about Inception"})
    assert first.status_code == 200
    assert first.json()["answer"] == answer

    replies.extend([parsed_query(), answer_response("Follow-up answer.")])
    assert client.post("/movies/chat", json={"message": "Follow up"}).status_code == 200
    assert requests[-2]["messages"][2] == {"role": "assistant", "content": answer[:2000]}


def test_history_is_cleared_on_restart(chat_client):
    client, replies, requests = chat_client
    application = create_app()
    application.dependency_overrides.update(client.app.dependency_overrides)
    for message in ["Before restart", "After restart"]:
        replies.extend([parsed_query(), answer_response("An answer.")])
        # Each lifespan represents a fresh server startup.
        with TestClient(application) as restarted_client:
            assert restarted_client.post("/movies/chat", json={"message": message}).status_code == 200
        assert requests[-2]["messages"][1:-1] == []


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
