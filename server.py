# app/server.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

from app.rag import rag_answer

app = FastAPI(title="Smart Support Agent", version="0.1.0")

# CORS (relax as needed for your UI origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to ["http://localhost:3000"] etc. for tighter control
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------- Schemas ----------
class Citation(BaseModel):
    id: int
    source: str
    preview: str

class AnswerResponse(BaseModel):
    answer: str
    citations: List[Citation]
    model: str

class QueryRequest(BaseModel):
    question: str = Field(..., description="User question")
    k: Optional[int] = Field(5, description="Top-K documents to retrieve")

# --------- Routes -----------
@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}

@app.post("/api/answer", response_model=AnswerResponse)
def answer(req: QueryRequest):
    out = rag_answer(req.question, k=req.k or 5)
    # pydantic-safe
    return AnswerResponse(
        answer=out["answer"],
        citations=[Citation(**c) for c in out["citations"]],
        model=out["model"],
    )