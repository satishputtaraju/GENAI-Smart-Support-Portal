# app/api.py
from time import perf_counter
from typing import List, Iterator

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from app.db import init_db, add_event, get_recent
from app.rag import rag_answer

# Optional streaming helper (if you added it in rag.py Step 4B)
try:
    from app.rag import stream_rag_answer  # yields str chunks
except Exception:
    stream_rag_answer = None  # fallback later

# ---------- App & CORS ----------
app = FastAPI(title="Smart Support Agent", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in prod
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize DB tables
init_db()

# ---------- Schemas ----------
class Citation(BaseModel):
    id: int
    source: str
    preview: str

class AnswerResponse(BaseModel):
    answer: str
    citations: List[Citation]
    model: str

class QueryRequest(BaseModel):
    question: str
    k: int = 5

# ---------- Routes ----------
@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.post("/api/answer", response_model=AnswerResponse)
def answer_api(req: QueryRequest):
    """
    Non-streaming RAG answer. Also logs to SQLite (support.db).
    """
    t0 = perf_counter()
    out = rag_answer(req.question, k=req.k or 5)
    latency = perf_counter() - t0

    # Gather sources for logging
    sources = [c.get("source", "") for c in out.get("citations", [])]

    # Persist event
    add_event(
        question=req.question,
        answer=out.get("answer", ""),
        model=out.get("model", ""),
        latency_s=latency,
        sources=sources,
    )

    return JSONResponse(out)

@app.post("/api/answer-stream")
def answer_stream(req: QueryRequest):
    """
    Streams the RAG answer as it’s generated.
    If streaming helper isn't available, falls back to single-chunk output.
    """
    def gen() -> Iterator[str]:
        if stream_rag_answer is not None:
            for chunk in stream_rag_answer(req.question, k=req.k or 5):
                yield chunk
        else:
            out = rag_answer(req.question, k=req.k or 5)
            yield out["answer"]

    # Plain text streaming works well with: curl -N ...
    return StreamingResponse(gen(), media_type="text/plain; charset=utf-8")

@app.get("/api/logs")
def recent_logs(limit: int = Query(20, ge=1, le=200)):
    """
    Small, read-only endpoint to inspect recent Q&A events.
    """
    rows = get_recent(limit)
    return JSONResponse([
        {
            "id": r.id,
            "ts": r.ts.isoformat() + "Z",
            "question": r.question,
            "model": r.model,
            "latency_s": r.latency_s,
            "sources": r.sources,
        }
        for r in rows
    ])