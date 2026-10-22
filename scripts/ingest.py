
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.database import DEFAULT_DB_PATH, ensure_tables

MOVIES_CSV = ROOT / "tmdb_5000_movies.csv"
CREDITS_CSV = ROOT / "tmdb_5000_credits.csv"

MOVIE_COLUMNS = [
    "budget",
    "genres",
    "homepage",
    "id",
    "keywords",
    "original_language",
    "original_title",
    "overview",
    "popularity",
    "production_companies",
    "production_countries",
    "release_date",
    "revenue",
    "runtime",
    "spoken_languages",
    "status",
    "tagline",
    "title",
    "vote_average",
    "vote_count",
]

CREDIT_COLUMNS = ["movie_id", "title", "cast", "crew"]


def parse_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def clean_text(value: str | None) -> str | None:
    if value is None or value == "":
        return None
    return value


def format_json(value: str | None) -> str | None:
    if value is None or value == "":
        return None
    try:
        return json.dumps(json.loads(value))
    except (TypeError, ValueError):
        return value


def load_movies_csv(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            normalized = {
                "budget": parse_int(row.get("budget")),
                "genres": format_json(row.get("genres")),
                "homepage": clean_text(row.get("homepage")),
                "id": parse_int(row.get("id")),
                "keywords": format_json(row.get("keywords")),
                "original_language": clean_text(row.get("original_language")),
                "original_title": clean_text(row.get("original_title")),
                "overview": clean_text(row.get("overview")),
                "popularity": parse_float(row.get("popularity")),
                "production_companies": format_json(row.get("production_companies")),
                "production_countries": format_json(row.get("production_countries")),
                "release_date": clean_text(row.get("release_date")),
                "revenue": parse_int(row.get("revenue")),
                "runtime": parse_float(row.get("runtime")),
                "spoken_languages": format_json(row.get("spoken_languages")),
                "status": clean_text(row.get("status")),
                "tagline": clean_text(row.get("tagline")),
                "title": clean_text(row.get("title")),
                "vote_average": parse_float(row.get("vote_average")),
                "vote_count": parse_int(row.get("vote_count")),
            }
            rows.append(normalized)
    return rows


def load_credits_csv(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            normalized = {
                "movie_id": parse_int(row.get("movie_id")),
                "title": clean_text(row.get("title")),
                "cast": format_json(row.get("cast")),
                "crew": format_json(row.get("crew")),
            }
            rows.append(normalized)
    return rows


def insert_movies(conn: sqlite3.Connection, rows: list[dict[str, object]]) -> int:
    insert_sql = (
        "INSERT OR REPLACE INTO movies ("
        + ", ".join(MOVIE_COLUMNS)
        + ") VALUES ("
        + ", ".join(["?" for _ in MOVIE_COLUMNS])
        + ")"
    )
    conn.executemany(
        insert_sql,
        [tuple(row.get(column) for column in MOVIE_COLUMNS) for row in rows],
    )
    conn.commit()
    return len(rows)


def insert_credits(conn: sqlite3.Connection, rows: list[dict[str, object]]) -> int:
    insert_sql = (
        "INSERT OR REPLACE INTO credits ("
        + ", ".join(CREDIT_COLUMNS)
        + ") VALUES ("
        + ", ".join(["?" for _ in CREDIT_COLUMNS])
        + ")"
    )
    conn.executemany(
        insert_sql,
        [tuple(row.get(column) for column in CREDIT_COLUMNS) for row in rows],
    )
    conn.commit()
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Load TMDB CSV files into a SQLite database.")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH, help="Destination SQLite DB path")
    parser.add_argument("--movies-csv", type=Path, default=MOVIES_CSV, help="Path to tmdb_5000_movies.csv")
    parser.add_argument("--credits-csv", type=Path, default=CREDITS_CSV, help="Path to tmdb_5000_credits.csv")
    args = parser.parse_args()

    conn = ensure_tables(args.db_path)
    try:
        movies_rows = load_movies_csv(args.movies_csv)
        credits_rows = load_credits_csv(args.credits_csv)

        inserted_movies = insert_movies(conn, movies_rows)
        inserted_credits = insert_credits(conn, credits_rows)

        print(f"Created database at: {args.db_path}")
        print(f"Inserted {inserted_movies} movies into movies table")
        print(f"Inserted {inserted_credits} credits into credits table")
    finally:
        conn.close()


if __name__ == "__main__":
    main()