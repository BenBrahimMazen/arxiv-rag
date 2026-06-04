"""Streamlit chat UI for the ArXiv Research Assistant."""
from __future__ import annotations

import json
import os
from collections.abc import Iterator

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

EXAMPLE_QUESTIONS = [
    "What methods reduce hallucination in large language models?",
    "How does LoRA reduce the cost of fine-tuning?",
    "What is FlashAttention and why is it faster?",
    "How does DPO differ from RLHF?",
    "What techniques improve transformer context length?",
]

st.set_page_config(page_title="ArXiv Research Assistant", page_icon="📚", layout="wide")


# ── API helpers ───────────────────────────────────────────────────────────
def fetch_health() -> dict:
    try:
        resp = requests.get(f"{API_URL}/health", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return {}


def stream_answer(question: str) -> Iterator[tuple[str, object]]:
    """Yield ('token', str) chunks then ('sources', list) from the SSE endpoint."""
    with requests.get(
        f"{API_URL}/query/stream",
        params={"question": question},
        stream=True,
        timeout=120,
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data = line[len("data: ") :]
            if data == "[DONE]":
                break
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "token":
                yield "token", event["data"]
            elif event.get("type") == "sources":
                yield "sources", event["data"]


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"📄 Sources ({len(sources)})"):
        for s in sources:
            title = s.get("title") or s["arxiv_id"]
            st.markdown(f"**[{title}]({s['url']})**  \n`arxiv:{s['arxiv_id']}`")


# ── session state ─────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []  # list[{role, content, sources}]
if "pending" not in st.session_state:
    st.session_state.pending = None

# ── sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Filters")
    st.slider("Year range", 2023, 2024, (2023, 2024), key="year_range")
    st.multiselect(
        "Categories", ["cs.LG", "cs.CV", "cs.CL"], default=["cs.LG"], key="categories"
    )
    st.slider("Sources to show", 1, 10, 5, key="num_sources")

    st.divider()
    health = fetch_health()
    st.subheader("📊 Index stats")
    if health:
        st.metric("Papers indexed", health.get("papers_indexed", 0))
        st.metric("Chunks indexed", health.get("chunks_indexed", 0))
        st.caption(f"API: {health.get('status', 'unknown')}")
    else:
        st.warning(f"API unreachable at {API_URL}")

# ── header ────────────────────────────────────────────────────────────────
st.title("📚 ArXiv Research Assistant")
st.caption(
    "Ask questions about ArXiv machine-learning papers — answers come with citations."
)

# ── example questions ─────────────────────────────────────────────────────
st.write("**Try an example:**")
cols = st.columns(len(EXAMPLE_QUESTIONS))
for col, example in zip(cols, EXAMPLE_QUESTIONS, strict=False):
    if col.button(example, use_container_width=True):
        st.session_state.pending = example

# ── render history ────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_sources(msg.get("sources", []))

# ── handle input ──────────────────────────────────────────────────────────
prompt = st.chat_input("Ask a question about ML papers...")
question = prompt or st.session_state.pending
st.session_state.pending = None

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        sources_box: list[dict] = []

        def token_stream() -> Iterator[str]:
            for kind, payload in stream_answer(question):
                if kind == "token":
                    yield payload
                elif kind == "sources":
                    sources_box.extend(payload)

        try:
            answer = st.write_stream(token_stream())
        except Exception as exc:  # noqa: BLE001
            answer = f"⚠️ Error contacting API: {exc}"
            st.error(answer)
        render_sources(sources_box)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources_box}
    )
