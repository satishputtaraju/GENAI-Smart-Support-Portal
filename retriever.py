# app/retriever.py
import os
from typing import List, Tuple

from langchain_ollama import OllamaEmbeddings
try:
    # Prefer the modern adapter
    from langchain_chroma import Chroma
except Exception:  # fallback if only community adapter is installed
    from langchain_community.vectorstores import Chroma  # type: ignore

# ---------- config (must match ingest.py) ----------
#CHROMA_DIR = os.getenv("CHROMA_DIR", "./.chroma")
#CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "support_docs")

CHROMA_DIR = os.getenv("CHROMA_DIR", "data/chroma")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "support_docs")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

TOP_K = int(os.getenv("RETRIEVER_TOP_K", "5"))
# ---------------------------------------------------


#def _embeddings():
#    return OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL)

def _embeddings():
    # IMPORTANT: must match ingest.py
    return OllamaEmbeddings(model="nomic-embed-text")
    # (no env var here, so we can't accidentally switch models)

def load_vectorstore() -> Chroma:
    """Open the persisted Chroma DB (read-only usage is fine)."""
    return Chroma(
        persist_directory=CHROMA_DIR,
        collection_name=CHROMA_COLLECTION,
        embedding_function=_embeddings(),
    )


def retrieve(query: str, k: int = TOP_K) -> List[Tuple[str, str]]:
    """Return (snippet, source_path) for top-k results."""
    vs = load_vectorstore()
    docs = vs.similarity_search(query, k=k)
    results = []
    for d in docs:
        text = (d.page_content or "").replace("\n", " ")
        results.append(
            (
                text[:240] + ("…" if len(text) > 240 else ""),
                d.metadata.get("source", ""),
            )
        )
    return results


# ----------------- quick CLI test -----------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print('Usage: python -m app.retriever "your question here"')
        sys.exit(1)
    q = " ".join(sys.argv[1:])
    print(f"\n[Q] {q}\n")
    try:
        hits = retrieve(q, k=TOP_K)
    except Exception as e:
        print(f"[ERROR] retrieval failed: {e}")
        raise
    if not hits:
        print("[INFO] no matches found.")
        sys.exit(0)

    for i, (snippet, src) in enumerate(hits, 1):
        print(f"{i}. {snippet}")
        print(f"   └─ source: {src}\n")