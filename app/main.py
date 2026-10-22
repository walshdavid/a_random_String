from fastapi import FastAPI

from app.repository import MovieRecord, MovieRepository
from app.schemas import MovieQuery

app = FastAPI(title="Movie Assistant", description="A movie recommendation and information API.", version="1.0.0")
repository = MovieRepository()


@app.post("/movies/search", response_model=list[MovieRecord])
def search_movies(query: MovieQuery) -> list[MovieRecord]:
    return repository.search(query)
