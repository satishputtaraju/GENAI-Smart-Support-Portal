# app/rag_stream.py
import os
from typing import Iterator, List, Dict, Any

from app.retriever import load_vectorstore
from app.rag import _build_prompt, TOP_K, LLM_MODEL, OLLAMA_BASE_URL  # reuse config/prompt

# Prefer ChatOllama (has .stream); fall back to community LLMs w/ manual chunking
try:
    from langchain_ollama import ChatOllama as _ChatOllama
except Exception:
    _ChatOllama = None

try:
    from langchain_community.llms import Ollama as _Ollama
except Exception:
    _Ollama = None


def _iter_chunks(text: str, n: int = 40) -> Iterator[str]:
    """Fallback: split a string into small chunks to mimic streaming."""
    for i in range(0, len(text), n):
        yield text[i : i + n]


def stream_rag_answer(question: str, k: int = TOP_K) -> Iterator[Dict[str, Any]]:
    """
    Generator that yields incremental pieces of the answer.
    Yields dicts like: {"type":"token","text":"..."} and final {"type":"done","citations":[...]}.
    """
    vs = load_vectorstore()
    docs = vs.as_retriever(search_kwargs={"k": k}).invoke(question)  # LC ≥0.2 style

    prompt = _build_prompt(question, docs)

    # Try true streaming via ChatOllama
    if _ChatOllama is not None:
        chat = _ChatOllama(
            base_url=OLLAMA_BASE_URL,
            model=LLM_MODEL,
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
        )
        for chunk in chat.stream(prompt):
            # chunk may be a BaseMessageChunk or str depending on version
            txt = getattr(chunk, "content", None)
            if txt is None and isinstance(chunk, str):
                txt = chunk
            if txt:
                yield {"type": "token", "text": txt}

    # Fallback: use non-streaming LLM and “fake” streaming
    elif _Ollama is not None:
        llm = _Ollama(
            base_url=OLLAMA_BASE_URL,
            model=LLM_MODEL,
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
        )
        # predict/invoke (version tolerant)
        try:
            full = llm.invoke(prompt)
        except Exception:
            full = llm.predict(prompt)
        for piece in _iter_chunks(full, 40):
            yield {"type": "token", "text": piece}

    else:
        raise RuntimeError(
            "No Ollama backend found for streaming. Install `langchain-ollama` for ChatOllama or keep the community LLM."
        )

    # Send citations once at the end
    citations = []
    for i, d in enumerate(docs, 1):
        citations.append(
            {
                "id": i,
                "source": d.metadata.get("source", ""),
                "preview": (d.page_content or "").replace("\n", " ")[:200] + "…",
            }
        )
    yield {"type": "done", "citations": citations, "model": LLM_MODEL}