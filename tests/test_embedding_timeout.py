"""OpenAI embedding requests get a timeout (GR_EMBEDDING_REQUEST_TIMEOUT_S)."""

from unittest.mock import patch

from gpt_researcher.memory.embeddings import Memory


def _kwargs(**embedding_kwargs):
    with patch("langchain_openai.OpenAIEmbeddings") as cls:
        Memory("openai", "text-embedding-3-small", **embedding_kwargs)
    return cls.call_args.kwargs


def test_default_timeout(monkeypatch):
    monkeypatch.delenv("GR_EMBEDDING_REQUEST_TIMEOUT_S", raising=False)
    assert _kwargs()["request_timeout"] == 120.0


def test_env_override_and_off(monkeypatch):
    monkeypatch.setenv("GR_EMBEDDING_REQUEST_TIMEOUT_S", "30")
    assert _kwargs()["request_timeout"] == 30.0
    monkeypatch.setenv("GR_EMBEDDING_REQUEST_TIMEOUT_S", "0")
    assert "request_timeout" not in _kwargs()


def test_explicit_kwarg_wins(monkeypatch):
    monkeypatch.delenv("GR_EMBEDDING_REQUEST_TIMEOUT_S", raising=False)
    assert _kwargs(timeout=5)["timeout"] == 5
    assert "request_timeout" not in _kwargs(timeout=5)
