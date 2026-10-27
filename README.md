# Movie Assistant

A FastAPI movie assistant using SQLite and a local Ollama or hosted Hugging Face model.

Requires Python 3.11+. Run commands from the project root.

## Setup

Download the [TMDB 5000 dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata/data)
and place `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` in the project root.
The ingestion commands below create or rebuild `tmdb.db`.

### Setup with uv

With uv installed:

```bash
uv sync
uv run python scripts/ingest.py
```

### Setup with pip

Alternatively, create and activate a virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r req.txt
python scripts/ingest.py
```

## Run locally (default)

Install [Ollama](https://ollama.com/download), then start it in a separate terminal:

```bash
ollama serve
```

Download the default model:

```bash
ollama pull gemma3:4b
```

Start the app using your chosen setup:

```bash
# uv
uv run python -m app.main

# pip, with the virtual environment activated
python -m app.main
```

To use another installed model, set `OLLAMA_MODEL` before starting the app,
for example `export OLLAMA_MODEL='qwen3:4b'`.

## Run with the hosted API

Pass the  --API flag to run against a Hugging Face hosted model. Set your Hugging Face token and a model available to your account that supports
chat and JSON output:

```bash
export HF_TOKEN='your-hugging-face-token'
export HF_MODEL='google/gemma-3-4b-it'
```

Start using your chosen setup

```bash
# uv
uv run python -m app.main --API

# pip, with the virtual environment activated
python -m app.main --API
```



Open http://127.0.0.1:8000/docs to try the API.
