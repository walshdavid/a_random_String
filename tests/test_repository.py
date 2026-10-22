import json
import sqlite3

import pytest

from app.database import init_db
from app.repository import MovieRepository
from app.schemas import MovieQuery


def _seed_db(db_path):
    conn = sqlite3.connect(db_path)
    init_db(conn)

    conn.execute(
        """
        INSERT INTO movies (
            budget, genres, homepage, id, keywords, original_language,
            original_title, overview, popularity, production_companies,
            production_countries, release_date, revenue, runtime,
            spoken_languages, status, tagline, title, vote_average, vote_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            1000000,
            json.dumps([
                {"id": 28, "name": "Action"},
                {"id": 10752, "name": "War"},
            ]),
            "",
            1,
            json.dumps([]),
            "en",
            "Dunkirk",
            "A wartime rescue mission.",
            7.4,
            json.dumps([]),
            json.dumps([]),
            "2017-07-21",
            500000000,
            106.0,
            json.dumps([{"iso_639_1": "en", "name": "English"}]),
            "Released",
            "Nothing can prepare you for this.",
            "Dunkirk",
            7.8,
            5000,
        ),
    )

    conn.execute(
        """
        INSERT INTO credits (movie_id, title, cast, crew)
        VALUES (?, ?, ?, ?)
        """,
        (
            1,
            "Dunkirk",
            json.dumps([]),
            json.dumps([
                {"id": 1, "name": "Christopher Nolan", "job": "Director"},
                {"id": 2, "name": "Christopher Nolan", "job": "Writer"},
            ]),
        ),
    )

    conn.commit()
    conn.close()


def test_repository_filters_title_genre_director_year_and_sorts(tmp_path):
    db_path = tmp_path / "movies.db"
    _seed_db(db_path)

    repo = MovieRepository(db_path)
    result = repo.search(
        MovieQuery(
            intent="search",
            title="Dunkirk",
            genre="War",
            director="Christopher Nolan",
            year=2017,
            sort_by="rating",
            limit=5,
        )
    )

    assert len(result) == 1
    assert result[0].title == "Dunkirk"
    assert result[0].director == "Christopher Nolan"
    assert "War" in result[0].genres
    assert result[0].release_year == 2017


def test_repository_limits_results(tmp_path):
    db_path = tmp_path / "movies.db"
    _seed_db(db_path)

    repo = MovieRepository(db_path)
    result = repo.search(MovieQuery(intent="search", limit=1))

    assert len(result) == 1


@pytest.mark.parametrize(
    ("sort_by", "expected_titles"),
    [
        (None, ["Low vote movie", "Boundary movie", "Dunkirk"]),
        ("rating", ["Dunkirk", "Boundary movie"]),
        ("popularity", ["Boundary movie", "Dunkirk"]),
    ],
)
def test_vote_threshold_applies_only_with_explicit_sort(
    tmp_path, sort_by, expected_titles
):
    db_path = tmp_path / "movies.db"
    _seed_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        conn.executemany(
            "INSERT INTO movies (id, title, vote_count, vote_average, popularity) "
            "VALUES (?, ?, ?, ?, ?)",
            [
                (2, "Low vote movie", 49, 10.0, 100.0),
                (3, "Boundary movie", 50, 7.0, 20.0),
            ],
        )
        conn.commit()
    finally:
        conn.close()

    result = MovieRepository(db_path).search(
        MovieQuery(intent="search", sort_by=sort_by)
    )

    assert [movie.title for movie in result] == expected_titles
