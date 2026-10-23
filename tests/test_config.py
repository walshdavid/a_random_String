import sys
from contextlib import contextmanager
from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock, Mock

import pytest
from fastapi.testclient import TestClient

from app import __main__ as cli
from app.model_config import LLMConfig
from app.main import create_app, get_llm_client
from app.query_parser import generate_answer, parse_query
from huggingface_hub import ChatCompletionOutput


def test_local_is_default_even_with_hosted_credentials(monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.setenv("HF_TOKEN", "hf_test_secret")
    monkeypatch.setenv("HF_MODEL", "example/hosted-model")
    config = LLMConfig.from_env()
    assert config.mode == "local"
    assert config.model == "qwen3:4b"
    assert config.token == "local"


@pytest.mark.parametrize("missing", ["HF_TOKEN", "HF_MODEL"])
def test_api_requires_credentials_and_model(monkeypatch, missing):
    monkeypatch.setenv("HF_TOKEN", "hf_test_secret")
    monkeypatch.setenv("HF_MODEL", "example/hosted-model")
    monkeypatch.delenv(missing)
    with pytest.raises(ValueError, match="HF_TOKEN and HF_MODEL"):
        LLMConfig.from_env(use_api=True)


def test_api_config_is_immutable_and_does_not_expose_token(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_test_secret")
    monkeypatch.setenv("HF_MODEL", "example/hosted-model")
    config = LLMConfig.from_env(use_api=True)
    assert config.mode == "api"
    assert config.base_url is None
    assert config.model == "example/hosted-model"
    assert "hf_test_secret" not in repr(config)
    with pytest.raises(FrozenInstanceError):
        config.model = "another-model"


def test_configuration_is_fixed_until_restart(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "original-model")
    application = create_app()
    with TestClient(application):
        original_config = application.state.llm_config
        monkeypatch.setenv("LLM_MODEL", "changed-model")
        assert application.state.llm_config is original_config
        assert application.state.llm_config.model == "original-model"
    with TestClient(application):
        assert application.state.llm_config.model == "changed-model"


def test_api_client_uses_hosted_routing(monkeypatch):
    constructor = MagicMock()
    monkeypatch.setattr("app.main.InferenceClient", constructor)
    config = LLMConfig(mode="api", model="example/model", base_url=None, token="hf_test")
    with contextmanager(get_llm_client)(config):
        pass
    constructor.assert_called_once_with(
        base_url=None, provider="auto", token="hf_test", timeout=30.0,
    )


def test_hosted_calls_use_selected_model_without_local_options(monkeypatch):
    # Actual SDK output objects, with no network calls or provider lookup.
    client = Mock()
    client.chat_completion.return_value = ChatCompletionOutput.parse_obj_as_instance({
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": '{"query": null}'}}],
        "created": 0, "id": "test", "model": "example/model",
    })
    config = LLMConfig(mode="api", model="example/model", base_url=None, token="hf_test")
    monkeypatch.setenv("LLM_MODEL", "must-not-be-read")
    assert parse_query(client, "A question", [], config) is None
    generate_answer(client, "A question", [], [], config)
    for call in client.chat_completion.call_args_list:
        assert call.kwargs["model"] == "example/model"
        assert call.kwargs["extra_body"] is None


@pytest.mark.parametrize("arguments, expected_mode", [([], "local"), (["--API"], "api")])
def test_launcher_selects_mode_once(monkeypatch, arguments, expected_mode):
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    monkeypatch.setenv("HF_MODEL", "example/model")
    monkeypatch.setattr(sys, "argv", ["app", *arguments])
    started = []

    def fake_run(application, **kwargs):
        with TestClient(application):
            started.append(application.state.llm_config.mode)
        assert kwargs == {"host": "127.0.0.1", "port": 8000}

    monkeypatch.setattr(cli.uvicorn, "run", fake_run)
    cli.main()
    assert started == [expected_mode]


def test_launcher_rejects_unconfigured_api_before_starting(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(sys, "argv", ["app", "--API"])
    server = Mock()
    monkeypatch.setattr(cli.uvicorn, "run", server)
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    server.assert_not_called()
