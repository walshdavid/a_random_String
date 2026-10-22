from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = ROOT / "tmdb.db"


def get_connection(db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute("DROP TABLE IF EXISTS credits")
    conn.execute("DROP TABLE IF EXISTS movies")

    conn.execute(
        """
        CREATE TABLE movies (
            budget INTEGER,
            genres TEXT,
            homepage TEXT,
            id INTEGER PRIMARY KEY,
            keywords TEXT,
            original_language TEXT,
            original_title TEXT,
            overview TEXT,
            popularity REAL,
            production_companies TEXT,
            production_countries TEXT,
            release_date TEXT,
            revenue INTEGER,
            runtime REAL,
            spoken_languages TEXT,
            status TEXT,
            tagline TEXT,
            title TEXT,
            vote_average REAL,
            vote_count INTEGER
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE credits (
            movie_id INTEGER PRIMARY KEY,
            title TEXT,
            cast TEXT,
            crew TEXT,
            FOREIGN KEY (movie_id) REFERENCES movies(id)
        )
        """
    )

    conn.commit()


def ensure_tables(db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    conn = get_connection(db_path)
    init_db(conn)
    return conn


def count_rows(conn: sqlite3.Connection, table_name: str) -> int:
    row = conn.execute(f"SELECT COUNT(*) AS count FROM {table_name}").fetchone()
    return int(row["count"])


def list_tables(conn: sqlite3.Connection) -> Iterable[str]:
    for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
    ):
        yield row[0]
