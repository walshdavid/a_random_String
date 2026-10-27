from __future__ import annotations

import argparse
import csv
import json
import sys
from contextlib import closing
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.database import DEFAULT_DB_PATH, get_connection, init_db

MOVIES_CSV = ROOT / "tmdb_5000_movies.csv"
CREDITS_CSV = ROOT / "tmdb_5000_credits.csv"


def load_movies_csv(path: Path) -> list[dict]:
    movies = []
    with path.open(encoding="utf-8", newline="") as csvfile:
        for row in csv.DictReader(csvfile):
            release_date = row.get("release_date") or ""
            movies.append({
                "id": int(row["id"]),
                "title": row.get("title") or None,
                "year": int(release_date[:4]) if release_date else None,
                "overview": row.get("overview") or None,
                "genres": [genre["name"] for genre in json.loads(row.get("genres") or "[]")],
                "keywords": json.dumps([
                    keyword["name"] for keyword in json.loads(row.get("keywords") or "[]")
                ]),
                "popularity": float(row["popularity"]) if row.get("popularity") else None,
                "vote_average": float(row["vote_average"]) if row.get("vote_average") else None,
                "vote_count": int(row["vote_count"]) if row.get("vote_count") else None,
            })
    return movies


def load_credits_csv(path: Path) -> dict[int, dict]:
    credits = {}
    with path.open(encoding="utf-8", newline="") as csvfile:
        for row in csv.DictReader(csvfile):
            cast = json.loads(row.get("cast") or "[]")
            crew = json.loads(row.get("crew") or "[]")
            credits[int(row["movie_id"])] = {
                "cast": json.dumps([actor["name"] for actor in cast if actor.get("name")]),
                "director": next(
                    (person["name"] for person in crew
                     if person.get("job") == "Director" and person.get("name")),
                    None,
                ),
            }
    return credits


def ingest(db_path: Path, movies_csv: Path, credits_csv: Path) -> int:
    # Read and validate both inputs before replacing the existing database tables.
    movies = load_movies_csv(movies_csv)
    credits = load_credits_csv(credits_csv)

    with closing(get_connection(db_path)) as conn, conn:
        # SQLite otherwise starts transactions only for writes, not schema changes.
        conn.execute("BEGIN")
        init_db(conn)
        for movie in movies:
            credit = credits.get(movie["id"], {"cast": "[]", "director": None})
            conn.execute(
                'INSERT INTO movies (id, title, year, overview, "cast", director, keywords, popularity) '
                'VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                (movie["id"], movie["title"], movie["year"], movie["overview"],
                 credit["cast"], credit["director"], movie["keywords"], movie["popularity"]),
            )
            conn.executemany(
                "INSERT INTO genres (movie_id, name) VALUES (?, ?)",
                [(movie["id"], name) for name in dict.fromkeys(movie["genres"])],
            )
            conn.execute(
                "INSERT INTO ratings (movie_id, vote_average, vote_count) VALUES (?, ?, ?)",
                (movie["id"], movie["vote_average"], movie["vote_count"]),
            )
    return len(movies)


def main() -> None:
    parser = argparse.ArgumentParser(description="Load TMDB CSV files into a SQLite database.")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH, help="Destination SQLite DB path")
    parser.add_argument("--movies-csv", type=Path, default=MOVIES_CSV, help="Path to tmdb_5000_movies.csv")
    parser.add_argument("--credits-csv", type=Path, default=CREDITS_CSV, help="Path to tmdb_5000_credits.csv")
    args = parser.parse_args()

    inserted_movies = ingest(args.db_path, args.movies_csv, args.credits_csv)
    print(f"Created database at: {args.db_path}")
    print(f"Inserted {inserted_movies} movies with genres, ratings and credits data")


if __name__ == "__main__":
    main()
