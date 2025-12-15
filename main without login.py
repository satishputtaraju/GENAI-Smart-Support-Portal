# app/main.py

from typing import Dict, Optional, List, Any
from datetime import datetime
from pathlib import Path
from uuid import uuid4
import inspect
import os
import io
import tempfile
from faster_whisper import WhisperModel

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.chains import build_chain
from app.ingest import router as ingest_router

# ---------- OpenAI / Whisper config ----------
from openai import OpenAI


local_whisper_model = WhisperModel("medium", device="cpu")  # or "cuda"


# IMPORTANT: this must be the *name* of the env var, not the key itself
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "whisper-1")

client: Optional[OpenAI] = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# -------------------------------------------------
# FastAPI app setup
# -------------------------------------------------
app = FastAPI(title="Smart Support Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # ok for local dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount existing /api/ingest router
app.include_router(ingest_router)

# -------------------------------------------------
# Whisper transcription models
# -------------------------------------------------
class TranscriptionResponse(BaseModel):
    text: str
    error: Optional[str] = None

@app.post("/api/transcribe", response_model=TranscriptionResponse)
async def transcribe(file: UploadFile = File(...)):
    try:
        raw = await file.read()

        # Save audio to temp file
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name

        # Run local transcription
        segments, info = local_whisper_model.transcribe(tmp_path)
        text = " ".join([seg.text for seg in segments]).strip()

        return TranscriptionResponse(text=text, error=None)

    except Exception as e:
        return TranscriptionResponse(text="", error=str(e))


# -------------------------------------------------
# Models & constants
# -------------------------------------------------
class ExactSearchRequest(BaseModel):
    text: str


class ExactSearchResponse(BaseModel):
    found: bool
    files: List[str]


SEARCH_PATHS = [
    "data/docs/dialogueText.csv",
    "data/docs/incident_utf8.txt",
]


class QueryRequest(BaseModel):
    question: str


class TranscriptionResponse(BaseModel):
    text: str
    error: Optional[str] = None


class AgentResponse(BaseModel):
    answer: str
    confidence: Optional[float] = None
    error: Optional[str] = None
    sources: List[Dict[str, Any]] = []
    # Optional extra field the UI can show if present
    chunk_count: Optional[int] = None


UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class UploadResponse(BaseModel):
    server_path: str
    original_filename: str


class LogEvent(BaseModel):
    event: str
    status: str
    file: Optional[str] = None
    ts: Optional[str] = None
    extra: Optional[Dict] = None


last_ingest_time: Optional[datetime] = None





# -------------------------------------------------
# Exact text search over raw files (optional helper)
# -------------------------------------------------
@app.post("/api/search_exact", response_model=ExactSearchResponse)
def search_exact(req: ExactSearchRequest):
    needle = req.text
    hits: List[str] = []

    for p in SEARCH_PATHS:
        path = Path(p)
        if not path.exists():
            continue

        txt = path.read_text(encoding="utf-8", errors="ignore")
        if needle in txt:
            hits.append(str(path))

    return ExactSearchResponse(found=bool(hits), files=hits)


# -------------------------------------------------
# QA / Answer endpoint
# -------------------------------------------------
agent = build_chain()  # async run(question) from chains.py


@app.post("/api/answer", response_model=AgentResponse)
async def api_answer(req: QueryRequest):
    try:
        call = (
            getattr(agent, "ainvoke", None)
            or getattr(agent, "invoke", None)
            or agent
        )

        # 1) LangChain-style chains with ainvoke/invoke
        if hasattr(call, "ainvoke"):
            result = await call.ainvoke(req.question)
        elif hasattr(call, "invoke"):
            result = call.invoke(req.question)
        else:
            # 2) Plain callable: handle sync vs async
            if inspect.iscoroutinefunction(call):
                result = await call(req.question)
            else:
                maybe = call(req.question)
                if inspect.iscoroutine(maybe):
                    result = await maybe
                else:
                    result = maybe

        # Normalize result
        if isinstance(result, BaseModel):
            result = result.model_dump()

        if isinstance(result, dict):
            return AgentResponse(
                answer=result.get("answer") or result.get("output") or "",
                confidence=result.get("confidence"),
                error=result.get("error"),
                sources=result.get("sources") or [],
                chunk_count=result.get("chunk_count"),
            )

        # Fallback: treat anything else as plain text
        return AgentResponse(answer=str(result), confidence=0.5, sources=[])

    except Exception as e:
        return AgentResponse(
            answer="",
            confidence=0.0,
            error=str(e),
            sources=[],
        )


# -------------------------------------------------
# File upload endpoint (used by the UI)
#   Step 1: upload file → server_path
#   Step 2 (frontend): call /api/ingest with that path
# -------------------------------------------------
@app.post("/api/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file has no name")

    suffix = Path(file.filename).suffix or ".txt"
    tmp_name = f"ui_{uuid4().hex}{suffix}"
    tmp_path = UPLOAD_DIR / tmp_name

    contents = await file.read()
    tmp_path.write_bytes(contents)

    return UploadResponse(
        server_path=str(tmp_path),
        original_filename=file.filename,
    )


# -------------------------------------------------
# Health check
# -------------------------------------------------
@app.get("/api/health")
def health():
    return {"status": "ok"}


# -------------------------------------------------
# Logging + ingest status helpers
# -------------------------------------------------
@app.post("/api/log")
def log(ev: LogEvent):
    print("LOG:", ev.model_dump())
    return {
        "ok": True,
        "received": ev.model_dump(),
        "server_ts": datetime.utcnow().isoformat() + "Z",
    }


@app.post("/api/ingest_done")
def ingest_done():
    global last_ingest_time
    last_ingest_time = datetime.utcnow()
    return {
        "status": "ok",
        "message": "Ingest completed",
        "ts": last_ingest_time,
    }


@app.get("/api/ingest_status")
def ingest_status():
    if last_ingest_time:
        return {
            "status": "completed",
            "last_run": last_ingest_time.isoformat() + "Z",
        }
    else:
        return {
            "status": "pending",
            "message": "Ingest not yet run",
        }
