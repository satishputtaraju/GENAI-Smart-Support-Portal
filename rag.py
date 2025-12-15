# app/rag.py
import os
from typing import List, Dict, Any

from app.retriever import load_vectorstore

# ---- LLM imports (support both community + chat backends) ----
LLM_BACKEND = "unknown"
try:
    # Classic text LLM
    from langchain_community.llms import Ollama as _Ollama
    LLM_BACKEND = "community_llms.Ollama"
except Exception:
    _Ollama = None

try:
    # Chat interface
    from langchain_ollama import ChatOllama as _ChatOllama
    LLM_BACKEND = "langchain_ollama.ChatOllama"
except Exception:
    _ChatOllama = None

# --------------------- CONFIG ---------------------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1:latest")      # e.g., "llama3.1" or "llama3.1:8b"
TOP_K = int(os.getenv("RETRIEVER_TOP_K", "5"))
MAX_CONTEXT_CHARS = int(os.getenv("RAG_MAX_CTX", "6000"))
# --------------------------------------------------


def _get_llm():
    """
    Return an object exposing .invoke(prompt:str)->str
    using either langchain_community.llms.Ollama or langchain_ollama.ChatOllama.
    """
    if _Ollama is not None:
        llm = _Ollama(
            base_url=OLLAMA_BASE_URL,
            model=LLM_MODEL,
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
        )

        class _InvokeWrapper:
            def __init__(self, llm):
                self.llm = llm
            def invoke(self, prompt: str) -> str:
                # Newer LC may have .invoke; older has .predict
                try:
                    return self.llm.invoke(prompt)
                except Exception:
                    return self.llm.predict(prompt)

        return _InvokeWrapper(llm)

    if _ChatOllama is not None:
        llm = _ChatOllama(
            base_url=OLLAMA_BASE_URL,
            model=LLM_MODEL,
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
        )

        class _ChatWrapper:
            def __init__(self, chat):
                self.chat = chat
            def invoke(self, prompt: str) -> str:
                out = self.chat.invoke(prompt)  # returns BaseMessage or str
                return getattr(out, "content", out)

        return _ChatWrapper(llm)

    raise RuntimeError(
        "No Ollama LLM backend found. Install `langchain-community` or `langchain-ollama`."
    )


def _build_prompt(question: str, docs: List[Dict[str, Any]]) -> str:
    """Make a structured prompt with citations."""
    header = (
        "You are a helpful support AI. Answer the user based ONLY on the context.\n"
        "If the answer is unknown from the context, say you don't know.\n"
        "Cite sources as [S1], [S2], … using the indices below.\n\n"
    )

    ctx_parts = []
    running = 0
    for i, d in enumerate(docs, 1):
        txt = (getattr(d, "page_content", "") or "").replace("\n", " ")
        meta = getattr(d, "metadata", {}) or {}
        src = meta.get("source", "")
        block = f"[S{i}] {txt}\n(source: {src})\n"
        if running + len(block) > MAX_CONTEXT_CHARS:
            break
        ctx_parts.append(block)
        running += len(block)

    context = "CONTEXT:\n" + ("\n".join(ctx_parts) if ctx_parts else "(no context)\n")
    user = f"\nQUESTION: {question}\n\nINSTRUCTIONS: Provide a concise answer first, then list citations as [S#]."
    return header + context + user


def _retrieve_docs(question: str, k: int, vs):
    """
    Version-tolerant retrieval:
      - LC ≤0.2.x: retriever.get_relevant_documents(q)
      - LC 0.3.x: retriever.invoke(q)
      - Last resort: vs.similarity_search(q, k=k)
    """
    retriever = vs.as_retriever(search_kwargs={"k": k})

    # Try old API
    try:
        return retriever.get_relevant_documents(question)
    except AttributeError:
        pass

    # Try new Runnable API
    try:
        return retriever.invoke(question)
    except Exception:
        pass

    # Final fallback
    try:
        return vs.similarity_search(question, k=k)
    except Exception:
        return []


def rag_answer(question: str, k: int = TOP_K) -> Dict[str, Any]:
    """
    End-to-end RAG:
      1) top-k retrieve
      2) build prompt with numbered snippets
      3) call LLM
      4) return text + citations
    """
    vs = load_vectorstore()
    docs = _retrieve_docs(question, k, vs)

    prompt = _build_prompt(question, docs)
    llm = _get_llm()
    answer_text = llm.invoke(prompt)

    citations = []
    for i, d in enumerate(docs, 1):
        meta = getattr(d, "metadata", {}) or {}
        preview = (getattr(d, "page_content", "") or "").replace("\n", " ")[:200] + "…"
        citations.append({"id": i, "source": meta.get("source", ""), "preview": preview})

    return {"answer": answer_text, "citations": citations, "model": LLM_MODEL}


# -------------- quick CLI test ----------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m app.rag \"your question here\"")
        raise SystemExit(1)
    q = " ".join(sys.argv[1:])
    out = rag_answer(q, k=TOP_K)
    print("\n=== ANSWER ===\n")
    print(out["answer"])
    print("\n=== CITATIONS ===")
    for c in out["citations"]:
        print(f"[S{c['id']}] {c['source']} — {c['preview']}")
    print(f"\n(model: {out['model']})")