"""LLM backends for generation: OpenAI (via LangChain) or an offline echo stub."""
from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Protocol

from src.config import Settings, get_settings
from src.logging_conf import get_logger

logger = get_logger(__name__)


class LLM(Protocol):
    """Minimal async LLM interface used by the RAG chain."""

    async def complete(self, system: str, user: str) -> str: ...

    def stream(self, system: str, user: str) -> AsyncGenerator[str, None]: ...


class OpenAIChatLLM:
    """GPT-4o-mini through ``langchain_openai.ChatOpenAI`` (temperature=0)."""

    def __init__(self, settings: Settings | None = None) -> None:
        from langchain_openai import ChatOpenAI

        self.settings = settings or get_settings()
        if not self.settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY required for openai LLM backend")
        self._llm = ChatOpenAI(
            model=self.settings.llm_model,
            temperature=0,
            api_key=self.settings.openai_api_key,
        )

    def _messages(self, system: str, user: str):
        from langchain_core.messages import HumanMessage, SystemMessage

        return [SystemMessage(content=system), HumanMessage(content=user)]

    async def complete(self, system: str, user: str) -> str:
        resp = await self._llm.ainvoke(self._messages(system, user))
        return resp.content

    async def stream(self, system: str, user: str) -> AsyncGenerator[str, None]:
        async for chunk in self._llm.astream(self._messages(system, user)):
            if chunk.content:
                yield chunk.content


class EchoLLM:
    """Deterministic offline LLM for tests/demos without an API key.

    Produces a grounded-looking answer by quoting the first context block and
    citing the papers present in the prompt, so the full pipeline is exercisable.
    """

    async def complete(self, system: str, user: str) -> str:
        return self._render(user)

    async def stream(self, system: str, user: str) -> AsyncGenerator[str, None]:
        for token in self._render(user).split(" "):
            yield token + " "

    @staticmethod
    def _render(user: str) -> str:
        import re

        ids = re.findall(r"arxiv:(\S+)", user)
        question = user.split("Question:")[-1].strip()
        if not ids:
            return "I could not find sufficient information in the retrieved papers."
        cites = " ".join(f"[arxiv:{i}]" for i in dict.fromkeys(ids))
        return (
            f"Based on the retrieved passages, here is a grounded summary relevant "
            f"to: {question} {cites}"
        )


def get_llm(settings: Settings | None = None) -> LLM:
    """Factory returning the configured LLM backend."""
    settings = settings or get_settings()
    if settings.llm_backend == "echo":
        logger.info("Using EchoLLM (offline stub)")
        return EchoLLM()
    return OpenAIChatLLM(settings)
