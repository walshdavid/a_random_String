import pytest
from fastapi.testclient import TestClient

from app.main import app, repository
from app.repository import MovieRecord
from app.schemas import MovieQuery


def test_search_returns_movies(monkeypatch):
    def fake_search(query):
        assert isinstance(query, MovieQuery)
        assert query.title == "Dunkirk"
        return [MovieRecord(1, "Dunkirk", None, ["War"], "Christopher Nolan", 2017, 7.4, 7.8)]

    monkeypatch.setattr(repository, "search", fake_search)
    with TestClient(app) as client:
        response = client.post(
            "/movies/search", json={"intent": "search", "title": "Dunkirk"}
        )

    assert response.status_code == 200
    assert response.json() == [
        {"id": 1, "title": "Dunkirk", "overview": None, "genres": ["War"],
         "director": "Christopher Nolan", "release_year": 2017,
         "popularity": 7.4, "rating": 7.8}
    ]


@pytest.mark.parametrize("payload", [{}, {"intent": "search", "limit": 21}])
def test_invalid_search_is_rejected_before_repository_call(monkeypatch, payload):
    def unexpected_search(query):
        pytest.fail("Invalid requests must not reach the repository")

    monkeypatch.setattr(repository, "search", unexpected_search)
    with TestClient(app) as client:
        response = client.post("/movies/search", json=payload)

    assert response.status_code == 422
