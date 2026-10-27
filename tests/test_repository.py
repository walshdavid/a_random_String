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
            id, title, year, overview, cast, director, keywords, popularity
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            1,
            "Dunkirk",
            2017,
            "A wartime rescue mission.",
            json.dumps(["Fionn Whitehead", "Harry Styles"]),
            "Christopher Nolan",
            json.dumps([]),
            7.4,
        ),
    )
    conn.executemany(
        "INSERT INTO genres (movie_id, name) VALUES (?, ?)",
        [(1, "Action"), (1, "War")],
    )
    conn.execute(
        "INSERT INTO ratings (movie_id, vote_average, vote_count) VALUES (?, ?, ?)",
        (1, 7.8, 5000),
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
            title="dUnKiRk",
            genre="wAr",
            director="nOlAn",
            year=2017,
            keywords=["rescue"],
            sort_by="rating",
            limit=5,
        )
    )

    assert len(result) == 1
    assert result[0].title == "Dunkirk"
    assert result[0].director == "Christopher Nolan"
    assert set(result[0].genres) == {"Action", "War"}
    assert result[0].release_year == 2017
    assert result[0].rating == 7.8
    assert result[0].cast == ["Fionn Whitehead", "Harry Styles"]


def test_multiple_genres_do_not_duplicate_movies(tmp_path):
    db_path = tmp_path / "movies.db"
    _seed_db(db_path)

    result = MovieRepository(db_path).search(MovieQuery(intent="search"))

    assert [movie.id for movie in result] == [1]
    assert set(result[0].genres) == {"Action", "War"}


def test_movies_without_ratings_or_genres_are_returned_without_explicit_sort(tmp_path):
    db_path = tmp_path / "movies.db"
    _seed_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute("INSERT INTO movies (id, title) VALUES (?, ?)", (2, "Unrated"))

    repo = MovieRepository(db_path)
    result = repo.search(MovieQuery(intent="search", title="Unrated"))

    assert len(result) == 1
    assert result[0].rating is None
    assert result[0].genres == []
    assert result[0].cast == []
    assert result[0].keywords == []
    assert repo.search(MovieQuery(intent="search", title="Unrated", sort_by="rating")) == []


@pytest.mark.parametrize("limit, expected_count", [(None, 20), (1, 1), (50, 50)])
def test_repository_limits_results(tmp_path, limit, expected_count):
    db_path = tmp_path / "movies.db"
    _seed_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            "INSERT INTO movies (id, title) VALUES (?, ?)",
            [(movie_id, f"Movie {movie_id}") for movie_id in range(2, 61)],
        )

    repo = MovieRepository(db_path)
    query = MovieQuery(intent="search") if limit is None else MovieQuery(intent="search", limit=limit)
    result = repo.search(query)

    assert len(result) == expected_count


@pytest.mark.parametrize(
    ("keywords", "genre", "limit", "expected_titles"),
    [
        (["  ChRiStMaS  "], None, 20, ["Holiday Drama", "Winter Comedy", "Family Reunion"]),
        (["christmas", "easter"], "Comedy", 20, ["Winter Comedy", "Spring Comedy", "Family Reunion"]),
        (["christmas"], "Comedy", 1, ["Winter Comedy"]),
        ([], "Comedy", 1, ["Popular Summer Comedy"]),
        (["missing"], None, 20, []),
        (["%"], None, 20, ["Literal Symbols"]),
        (["_"], None, 20, ["Literal Symbols"]),
        (["!"], None, 20, ["Literal Symbols"]),
        (["%' OR 1=1 --"], None, 20, []),
        (["999"], None, 20, []),
    ],
)
def test_keyword_search_matches_overview_or_metadata_with_other_filters(
    tmp_path, keywords, genre, limit, expected_titles
):
    db_path = tmp_path / "movies.db"
    _seed_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        conn.executemany(
            "INSERT INTO movies (id, title, overview, keywords, popularity) "
            "VALUES (?, ?, ?, ?, ?)",
            [
                (2, "Winter Comedy", "A Christmas adventure.", "[]", 20),
                (3, "Family Reunion", None, '["christmas"]', 10),
                (4, "Holiday Drama", "A family gathering.", '["christmas"]', 30),
                (5, "Spring Comedy", "An Easter adventure.", "[]", 15),
                (6, "Popular Summer Comedy", "Friends visit a beach.", "[]", 1000),
                (7, "Literal Symbols", "100% fun with an_under_score!", "[]", 5),
                (8, "Missing Overview", None, "[]", 50),
            ],
        )
        conn.executemany(
            "INSERT INTO genres (movie_id, name) VALUES (?, ?)",
            [(2, "Comedy"), (3, "Comedy"), (4, "Drama"), (5, "Comedy"),
             (6, "Comedy"), (7, "Comedy"), (8, "Comedy")],
        )
        conn.commit()
    finally:
        conn.close()

    results = MovieRepository(db_path).search(
        MovieQuery(intent="recommend", keywords=keywords, genre=genre, limit=limit)
    )

    assert [movie.title for movie in results] == expected_titles
    for movie in results:
        if movie.title == "Family Reunion":
            assert movie.keywords == ["christmas"]


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
            "INSERT INTO movies (id, title, popularity) VALUES (?, ?, ?)",
            [
                (2, "Low vote movie", 100.0),
                (3, "Boundary movie", 20.0),
            ],
        )
        conn.executemany(
            "INSERT INTO ratings (movie_id, vote_count, vote_average) VALUES (?, ?, ?)",
            [(2, 49, 10.0), (3, 50, 7.0)],
        )
        conn.commit()
    finally:
        conn.close()

    result = MovieRepository(db_path).search(
        MovieQuery(intent="search", sort_by=sort_by)
    )

    assert [movie.title for movie in result] == expected_titles
