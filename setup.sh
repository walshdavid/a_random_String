#!/usr/bin/env bash
set -e

cd -- "$(dirname -- "$0")"

if [[ ! -f tmdb_5000_movies.csv || ! -f tmdb_5000_credits.csv ]]; then
    echo "Download both TMDB CSV files into the project root first (see README.md)." >&2
    exit 1
fi

# Install and start Ollama separately, as described in README.md.
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python scripts/ingest.py
ollama pull "${OLLAMA_MODEL:-gemma3:4b}"
python -m app.main
