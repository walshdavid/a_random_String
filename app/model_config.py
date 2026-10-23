"""Configuration is read once when starting the application."""

import os
from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class LLMConfig:
    mode: Literal["local", "api"]
    model: str
    base_url: str | None
    token: str = field(repr=False)

    @classmethod
    def from_env(cls, *, use_api: bool = False) -> "LLMConfig":
        if use_api:
            token = os.getenv("HF_TOKEN", "").strip()
            model = os.getenv("HF_MODEL", "").strip()
            if not token or not model:
                raise ValueError("API mode requires HF_TOKEN and HF_MODEL")
            return cls(mode="api", model=model, base_url=None, token=token)

        return cls(
            mode="local",
            model=os.getenv("LLM_MODEL", "").strip() or "qwen3:4b",
            base_url=os.getenv("LLM_BASE_URL", "").strip() or "http://127.0.0.1:11434",
            token="local",
        )
