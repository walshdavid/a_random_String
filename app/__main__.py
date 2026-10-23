"""Start with `uv run python -m app` or add `--API` for hosted inference."""

import argparse

import uvicorn

from app.model_config import LLMConfig
from app.main import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the movie assistant")
    parser.add_argument("--API", action="store_true", help="Use Hugging Face hosted inference")
    args = parser.parse_args()
    try:
        config = LLMConfig.from_env(use_api=args.API)
    except ValueError as exc:
        parser.error(str(exc))

    print(f"Starting in {config.mode} mode using {config.model}")
    uvicorn.run(create_app(config), host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
