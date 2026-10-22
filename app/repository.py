from __future__ import annotations

import json
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.database import DEFAULT_DB_PATH, get_connection
from app.schemas import MovieQuery


@dataclass(frozen=True)
class MovieRecord:
    id: int
    title: str
    overview: str | None
    genres: list[str]
    director: str | None
    release_year: int | None
    popularity: float | None
    rating: float | None

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "MovieRecord":
        genres = []
        raw_genres = row.get("genres")
        if raw_genres:
            try:
                parsed = json.loads(raw_genres)
            except (TypeError, ValueError):
                parsed = []
            if isinstance(parsed, list):
                genres = [item.get("name", "") for item in parsed if isinstance(item, dict) and item.get("name")]

        return cls(
            id=int(row["id"]),
            title=row["title"] or "",
            overview=row.get("overview"),
            genres=genres,
            director=row.get("director"),
            release_year=row.get("release_year"),
            popularity=row.get("popularity"),
            rating=row.get("vote_average"),
        )


class MovieRepository:
    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)

    def search(self, query: MovieQuery) -> list[MovieRecord]:
        if isinstance(query, dict):
            query = MovieQuery.model_validate(query)
        if not isinstance(query, MovieQuery):
            raise TypeError("query must be a MovieQuery instance or mapping")

        conditions: list[str] = []
        params: list[Any] = []

        if query.title:
            conditions.append("LOWER(m.title) LIKE LOWER(?)")
            params.append(f"%{query.title}%")

        if query.genre:
            conditions.append(
                "EXISTS ("
                "SELECT 1 FROM json_each(m.genres) AS genre_item "
                "WHERE LOWER(json_extract(genre_item.value, '$.name')) = LOWER(?)"
                ")"
            )
            params.append(query.genre)

        if query.director:
            conditions.append(
                "EXISTS ("
                "SELECT 1 FROM json_each(COALESCE(c.crew, '[]')) AS crew_item "
                "WHERE json_extract(crew_item.value, '$.job') = 'Director' "
                "AND LOWER(json_extract(crew_item.value, '$.name')) LIKE LOWER(?)"
                ")"
            )
            params.append(f"%{query.director}%")

        if query.year is not None:
            conditions.append("CAST(strftime('%Y', m.release_date) AS INTEGER) = ?")
            params.append(query.year)

        base_sql = (
            "SELECT m.id, m.title, m.overview, m.genres, m.popularity, m.vote_average, "
            "m.release_date, CAST(strftime('%Y', m.release_date) AS INTEGER) AS release_year, "
            "COALESCE((SELECT json_extract(crew_item.value, '$.name') "
            "FROM json_each(COALESCE(c.crew, '[]')) AS crew_item "
            "WHERE json_extract(crew_item.value, '$.job') = 'Director' LIMIT 1), '') AS director "
            "FROM movies AS m "
            "LEFT JOIN credits AS c ON c.movie_id = m.id"
        )

        # Add minimum vote count filter
        if query.sort_by:
            conditions.append("m.vote_count >= ?")
            params.append(50)

        if conditions:
            base_sql += " WHERE " + " AND ".join(conditions)

        if query.sort_by == "rating":
            order_by = "ORDER BY m.vote_average DESC, m.popularity DESC"
        elif query.sort_by == "popularity":
            order_by = "ORDER BY m.popularity DESC, m.vote_average DESC"
        else:
            order_by = "ORDER BY m.popularity DESC, m.vote_average DESC"

        sql = f"{base_sql} {order_by} LIMIT ?"
        params.append(query.limit)

        with closing(get_connection(self.db_path)) as conn:
            rows = conn.execute(sql, params).fetchall()

        return [MovieRecord.from_row(dict(row)) for row in rows]
