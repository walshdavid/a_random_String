import csv
import json
import sqlite3
from contextlib import closing

import pytest

from app.database import get_connection, init_db
from scripts.ingest import ingest


def _write_csv(path, fieldnames, rows):
    with path.open("w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def csv_inputs(tmp_path):
    movies_csv = tmp_path / "movies.csv"
    credits_csv = tmp_path / "credits.csv"
    movies = [{
        "id": "27205",
        "title": "Inception",
        "release_date": "2010-07-16",
        "overview": "A thief enters dreams.",
        "genres": json.dumps([{"id": 28, "name": "Action"},
                              {"id": 878, "name": "Science Fiction"},
                              {"id": 28, "name": "Action"}]),
        "keywords": json.dumps([{"id": 1, "name": "dream"}]),
        "popularity": "167.58371",
        "vote_average": "8.1",
        "vote_count": "13752",
    }, {
        "id": "603",
        "title": "Missing details",
    }]
    credits = [{
        "movie_id": "27205",
        "title": "A title that does not match",
        "cast": json.dumps([{"id": 6193, "name": "Leonardo DiCaprio", "character": "Cobb"},
                            {"id": 24045, "name": "Joseph Gordon-Levitt", "character": "Arthur"}]),
        "crew": json.dumps([{"name": "A writer", "job": "Writer"},
                            {"name": "Christopher Nolan", "job": "Director"},
                            {"name": "Another director", "job": "Director"}]),
    }]
    _write_csv(movies_csv, list(movies[0]), movies)
    _write_csv(credits_csv, list(credits[0]), credits)
    return movies_csv, credits_csv


def test_ingestion_normalizes_data_and_replaces_legacy_tables(tmp_path, csv_inputs):
    db_path = tmp_path / "movies.db"
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute("CREATE TABLE movies (id INTEGER PRIMARY KEY, title TEXT)")
        conn.execute("CREATE TABLE credits (movie_id INTEGER PRIMARY KEY)")

    assert ingest(db_path, *csv_inputs) == 2

    with closing(get_connection(db_path)) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        assert tables == {"movies", "genres", "ratings"}
        assert [row[0] for row in conn.execute("SELECT id FROM movies ORDER BY id")] == [603, 27205]

        movie = conn.execute("SELECT * FROM movies WHERE id = 27205").fetchone()
        assert movie["title"] == "Inception"
        assert movie["year"] == 2010
        assert movie["overview"] == "A thief enters dreams."
        assert json.loads(movie["cast"]) == ["Leonardo DiCaprio", "Joseph Gordon-Levitt"]
        assert movie["director"] == "Christopher Nolan"
        assert json.loads(movie["keywords"]) == ["dream"]
        assert movie["popularity"] == 167.58371
        assert [row[0] for row in conn.execute(
            "SELECT name FROM genres WHERE movie_id = 27205 ORDER BY name"
        )] == ["Action", "Science Fiction"]
        assert tuple(conn.execute("SELECT * FROM ratings WHERE movie_id = 27205").fetchone()) == (
            27205, 8.1, 13752,
        )

        missing = conn.execute("SELECT * FROM movies WHERE id = 603").fetchone()
        assert missing["year"] is None
        assert missing["overview"] is None
        assert json.loads(missing["cast"]) == []
        assert missing["director"] is None
        assert json.loads(missing["keywords"]) == []
        assert missing["popularity"] is None
        assert conn.execute("SELECT COUNT(*) FROM genres WHERE movie_id = 603").fetchone()[0] == 0
        assert tuple(conn.execute("SELECT * FROM ratings WHERE movie_id = 603").fetchone()) == (
            603, None, None,
        )
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_empty_cast_and_crew_are_supported(tmp_path, csv_inputs):
    movies_csv, credits_csv = csv_inputs
    _write_csv(credits_csv, ["movie_id", "cast", "crew"], [{"movie_id": "27205"}])
    db_path = tmp_path / "movies.db"

    ingest(db_path, movies_csv, credits_csv)

    with closing(get_connection(db_path)) as conn:
        movie = conn.execute('SELECT "cast", director FROM movies WHERE id = 27205').fetchone()
        assert json.loads(movie["cast"]) == []
        assert movie["director"] is None


@pytest.mark.parametrize("failure", ["invalid_json", "duplicate_id"])
def test_failed_ingestion_preserves_existing_database(tmp_path, csv_inputs, failure):
    movies_csv, credits_csv = csv_inputs
    db_path = tmp_path / "movies.db"
    with closing(sqlite3.connect(db_path)) as conn, conn:
        conn.execute("CREATE TABLE movies (id INTEGER PRIMARY KEY, title TEXT)")
        conn.execute("INSERT INTO movies VALUES (99, 'Keep this movie')")
        conn.execute("CREATE TABLE credits (movie_id INTEGER PRIMARY KEY)")
        conn.execute("INSERT INTO credits VALUES (99)")

    with movies_csv.open(encoding="utf-8", newline="") as csvfile:
        movies = list(csv.DictReader(csvfile))
    if failure == "invalid_json":
        movies[0]["genres"] = "broken JSON"
        expected_error = json.JSONDecodeError
    else:
        movies.append(movies[0])
        expected_error = sqlite3.IntegrityError
    _write_csv(movies_csv, list(movies[0]), movies)

    with pytest.raises(expected_error):
        ingest(db_path, movies_csv, credits_csv)

    with closing(get_connection(db_path)) as conn:
        assert tuple(conn.execute("SELECT * FROM movies").fetchone()) == (99, "Keep this movie")
        assert conn.execute("SELECT movie_id FROM credits").fetchone()[0] == 99
        assert {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )} == {"movies", "credits"}


@pytest.mark.parametrize("statement", [
    "INSERT INTO genres VALUES (1, 'Action')",
    "INSERT INTO genres VALUES (999, 'Action')",
    "INSERT INTO ratings VALUES (1, 7.0, 10)",
    "INSERT INTO ratings VALUES (999, 7.0, 10)",
])
def test_genres_and_ratings_enforce_keys(tmp_path, statement):
    with closing(get_connection(tmp_path / "movies.db")) as conn:
        init_db(conn)
        conn.execute("INSERT INTO movies (id, title) VALUES (1, 'Movie')")
        conn.execute("INSERT INTO genres VALUES (1, 'Action')")
        conn.execute("INSERT INTO ratings VALUES (1, 8.0, 100)")
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(statement)
