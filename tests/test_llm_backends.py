"""Tests for LLM backend selection and the echo stub (no network)."""
from __future__ import annotations

import pytest

from src.config import Settings
from src.generation.llm import EchoLLM, _openai_compatible_config, get_llm


def test_echo_backend_selected():
    llm = get_llm(Settings(llm_backend="echo"))
    assert isinstance(llm, EchoLLM)


def test_groq_config_resolves():
    settings = Settings(llm_backend="groq", groq_api_key="gsk-test")
    model, api_key, base_url = _openai_compatible_config(settings)
    assert model == settings.groq_model
    assert api_key == "gsk-test"
    assert "groq.com" in base_url


def test_groq_requires_key():
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        _openai_compatible_config(Settings(llm_backend="groq", groq_api_key=None))


def test_openai_config_has_no_base_url():
    settings = Settings(llm_backend="openai", openai_api_key="sk-test")
    model, api_key, base_url = _openai_compatible_config(settings)
    assert api_key == "sk-test"
    assert base_url is None


@pytest.mark.asyncio
async def test_echo_streams_and_cites():
    llm = EchoLLM()
    prompt = "[1] arxiv:2301.00001 (section: Body)\ntext\n\nQuestion: what?"
    tokens = [t async for t in llm.stream("sys", prompt)]
    out = "".join(tokens)
    assert "arxiv:2301.00001" in out
