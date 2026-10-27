from __future__ import annotations

import json
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.database import DEFAULT_DB_PATH, get_connection
from app.schemas import MovieQuery


def _json_strings(raw: str | None) -> list[str]:
    try:
        items = json.loads(raw) if raw else []
    except (TypeError, ValueError):
        return []
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, str) and item]


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
    keywords: list[str] = field(default_factory=list)
    cast: list[str] = field(default_factory=list)

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "MovieRecord":
        return cls(
            id=int(row["id"]),
            title=row["title"] or "",
            overview=row.get("overview"),
            genres=_json_strings(row.get("genres")),
            director=row.get("director"),
            release_year=row.get("release_year"),
            popularity=row.get("popularity"),
            rating=row.get("vote_average"),
            keywords=_json_strings(row.get("keywords")),
            cast=_json_strings(row.get("cast")),
        )


class MovieRepository:
    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)

    def search(self, query: MovieQuery) -> list[MovieRecord]:
        if isinstance(query, dict):
            query = MovieQuery.model_validate(query)
        if not isinstance(query, MovieQuery):
            raise TypeError("query must be a MovieQuery instance or mapping")
        if not query.has_criteria:
            return []

        conditions: list[str] = []
        params: list[Any] = []

        if query.title:
            conditions.append("LOWER(m.title) LIKE LOWER(?)")
            params.append(f"%{query.title}%")

        if query.genre:
            conditions.append(
                "EXISTS ("
                "SELECT 1 FROM genres AS g "
                "WHERE g.movie_id = m.id AND LOWER(g.name) = LOWER(?)"
                ")"
            )
            params.append(query.genre)

        if query.director:
            conditions.append("LOWER(m.director) LIKE LOWER(?)")
            params.append(f"%{query.director}%")

        if query.cast:
            escaped = query.cast.replace("!", "!!").replace("%", "!%").replace("_", "!_")
            conditions.append(
                "EXISTS ("
                'SELECT 1 FROM json_each(m."cast") AS cast_member '
                "WHERE LOWER(cast_member.value) LIKE LOWER(?) ESCAPE '!')"
            )
            params.append(f"%{escaped}%")

        if query.year is not None:
            conditions.append("m.year = ?")
            params.append(query.year)

        if query.keywords:
            keyword_conditions = []
            for keyword in query.keywords:
                # Search literal text; SQL LIKE wildcards in a keyword are escaped.
                escaped = keyword.replace("!", "!!").replace("%", "!%").replace("_", "!_")
                keyword_conditions.append(
                    "(LOWER(m.overview) LIKE LOWER(?) ESCAPE '!' OR EXISTS ("
                    "SELECT 1 FROM json_each(COALESCE(m.keywords, '[]')) AS keyword_item "
                    "WHERE LOWER(keyword_item.value) LIKE LOWER(?) ESCAPE '!'))"
                )
                params.extend([f"%{escaped}%", f"%{escaped}%"])
            conditions.append("(" + " OR ".join(keyword_conditions) + ")")

        base_sql = (
            'SELECT m.id, m.title, m.year AS release_year, m.overview, m."cast", '
            "m.director, m.keywords, m.popularity, r.vote_average, "
            "(SELECT json_group_array(g.name) FROM genres AS g "
            "WHERE g.movie_id = m.id) AS genres "
            "FROM movies AS m "
            "LEFT JOIN ratings AS r ON r.movie_id = m.id"
        )

        # Add minimum vote count filter
        if query.sort_by:
            conditions.append("r.vote_count >= ?")
            params.append(50)

        if conditions:
            base_sql += " WHERE " + " AND ".join(conditions)

        if query.sort_by == "rating":
            order_by = "ORDER BY r.vote_average DESC, m.popularity DESC"
        else:
            order_by = "ORDER BY m.popularity DESC, r.vote_average DESC"

        sql = f"{base_sql} {order_by} LIMIT ?"
        params.append(query.limit)

        with closing(get_connection(self.db_path)) as conn:
            rows = conn.execute(sql, params).fetchall()

        return [MovieRecord.from_row(dict(row)) for row in rows]
