from __future__ import annotations

import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = ROOT / "tmdb.db"


def get_connection(db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Recreate the schema; the caller owns the transaction."""
    conn.execute("DROP TABLE IF EXISTS genres")
    conn.execute("DROP TABLE IF EXISTS ratings")
    conn.execute("DROP TABLE IF EXISTS credits")
    conn.execute("DROP TABLE IF EXISTS movies")

    conn.execute(
        """
        CREATE TABLE movies (
            id INTEGER PRIMARY KEY,
            title TEXT,
            year INTEGER,
            overview TEXT,
            "cast" TEXT NOT NULL DEFAULT '[]',
            director TEXT,
            keywords TEXT NOT NULL DEFAULT '[]',
            popularity REAL
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE genres (
            movie_id INTEGER NOT NULL REFERENCES movies(id),
            name TEXT NOT NULL,
            PRIMARY KEY (movie_id, name)
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE ratings (
            movie_id INTEGER PRIMARY KEY REFERENCES movies(id),
            vote_average REAL,
            vote_count INTEGER
        )
        """
    )
