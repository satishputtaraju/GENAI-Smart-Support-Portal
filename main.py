# app/main.py

from typing import Dict, Optional, List, Any, Set
from datetime import datetime
from pathlib import Path
from uuid import uuid4
import inspect
import os
import json
import tempfile
import logging
import sqlite3
from fastapi.responses import FileResponse

from faster_whisper import WhisperModel

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    Depends,
    Header,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.chains import build_chain
from app.ingest import router as ingest_router

# ---------- Logging config ----------
logging.basicConfig(
    filename="support_agent.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("support_agent")

# Optional: quiet down very chatty libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)


# ---------- OpenAI / Whisper config ----------
from openai import OpenAI  # kept in case you later use OpenAI LLMs

# Local Whisper model (Faster-Whisper)
local_whisper_model = WhisperModel("medium", device="cpu")  # or "cuda"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "whisper-1")

client: Optional[OpenAI] = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# ---------- Simple user store (Sign Up / Login) ----------
USERS_FILE = Path("data/users.json")
USERS: Dict[str, str] = {}  # email(lowercased) -> password (plain text, dev only!)
ACTIVE_TOKENS: Set[str] = set()

# ---------- SQLite log DB ----------
LOG_DB_PATH = Path("data/logs.sqlite3")
LOG_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

log_conn = sqlite3.connect(LOG_DB_PATH, check_same_thread=False)
log_conn.execute(
    """
    CREATE TABLE IF NOT EXISTS logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,
        event TEXT NOT NULL,
        status TEXT,
        file TEXT,
        extra_json TEXT
    )
    """
)
log_conn.commit()


def insert_log_event(ev: "LogEvent") -> None:
    """Persist a frontend/backend log event to SQLite."""
    ts = ev.ts or datetime.utcnow().isoformat() + "Z"
    extra_json = json.dumps(ev.extra or {}, ensure_ascii=False)
    log_conn.execute(
        """
        INSERT INTO logs (ts, event, status, file, extra_json)
        VALUES (?, ?, ?, ?, ?)
        """,
        (ts, ev.event, ev.status, ev.file, extra_json),
    )
    log_conn.commit()

def load_users() -> None:
    global USERS
    if USERS_FILE.exists():
        try:
            USERS = json.loads(USERS_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            logger.exception("Failed to load users.json: %s", e)
            USERS = {}
    else:
        USERS = {}

    # Optionally seed an admin user from env vars (for convenience)
    admin_email = os.getenv("SSA_USERNAME")
    admin_pass = os.getenv("SSA_PASSWORD")
    if admin_email and admin_pass:
        key = admin_email.strip().lower()
        if key not in USERS:
            USERS[key] = admin_pass
            save_users()
            logger.info("Seeded admin user from env: %s", key)


def save_users() -> None:
    USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    USERS_FILE.write_text(json.dumps(USERS, indent=2), encoding="utf-8")


load_users()


# ---------- Auth helpers ----------
def require_auth(x_auth_token: str = Header(default=None)):
    """
    Simple auth guard. Expects a header 'X-Auth-Token' containing
    a token previously issued by /api/login or /api/signup.
    """
    if not x_auth_token or x_auth_token not in ACTIVE_TOKENS:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return True










# -------------------------------------------------
# FastAPI app setup
# -------------------------------------------------
app = FastAPI(title="Smart Support Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ok for local dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount existing /api/ingest router (kept as-is)
app.include_router(ingest_router)

# -------------------------------------------------
# Pydantic models
# -------------------------------------------------
class SignupRequest(BaseModel):
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class LoginResponse(BaseModel):
    token: str


class TranscriptionResponse(BaseModel):
    text: str
    error: Optional[str] = None


class ExactSearchRequest(BaseModel):
    text: str


class ExactSearchResponse(BaseModel):
    found: bool
    files: List[str]


class QueryRequest(BaseModel):
    question: str


class AgentResponse(BaseModel):
    answer: str
    confidence: Optional[float] = None
    error: Optional[str] = None
    sources: List[Dict[str, Any]] = []
    chunk_count: Optional[int] = None  # Optional extra field the UI can show if present


# UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR = Path("data/docs")
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

SEARCH_PATHS = [
    "data/docs/dialogueText.csv",
    "data/docs/incident_utf8.txt",
]

# -------------------------------------------------
# Auth endpoints (Sign Up / Login / Logout)
# -------------------------------------------------
@app.post("/api/signup", response_model=LoginResponse)
def signup(req: SignupRequest):
    email = req.email.strip().lower()
    if not email or not req.password:
        raise HTTPException(status_code=400, detail="Email and password are required.")

    if email in USERS:
        raise HTTPException(status_code=400, detail="User already exists.")

    USERS[email] = req.password
    save_users()

    token = uuid4().hex
    ACTIVE_TOKENS.add(token)
    logger.info("signup email=%s token_issued=%s", email, token[:8])
    return LoginResponse(token=token)


@app.post("/api/login", response_model=LoginResponse)
def login(req: LoginRequest):
    email = req.email.strip().lower()
    password = req.password

    if email not in USERS or USERS[email] != password:
        logger.warning("login_failed email=%s", email)
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    token = uuid4().hex
    ACTIVE_TOKENS.add(token)
    logger.info("login_success email=%s token_issued=%s", email, token[:8])
    return LoginResponse(token=token)


@app.post("/api/logout")
def logout(x_auth_token: str = Header(default=None)):
    if x_auth_token in ACTIVE_TOKENS:
        ACTIVE_TOKENS.remove(x_auth_token)
        logger.info("logout token=%s", x_auth_token[:8])
    return {"status": "logged_out"}


# -------------------------------------------------
# Whisper transcription (local Faster-Whisper)
# -------------------------------------------------
@app.post("/api/transcribe", response_model=TranscriptionResponse)
async def transcribe(
    file: UploadFile = File(...),
    auth: bool = Depends(require_auth),
):
    try:
        raw = await file.read()

        # Save audio to temp file
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name

        logger.info("transcribe_start tmp_path=%s size_bytes=%d", tmp_path, len(raw))

        # Run local transcription
        segments, info = local_whisper_model.transcribe(tmp_path)
        text = " ".join([seg.text for seg in segments]).strip()

        logger.info(
            "transcribe_done language=%s duration=%s text_preview=%r",
            getattr(info, "language", "unknown"),
            getattr(info, "duration", "unknown"),
            text[:80],
        )

        return TranscriptionResponse(text=text, error=None)

    except Exception as e:
        logger.exception("transcribe_failed: %s", e)
        return TranscriptionResponse(text="", error=str(e))


# -------------------------------------------------
# Exact text search over raw files (optional helper)
# -------------------------------------------------
@app.post("/api/search_exact", response_model=ExactSearchResponse)
def search_exact(
    req: ExactSearchRequest,
    auth: bool = Depends(require_auth),
):
    needle = req.text
    hits: List[str] = []

    logger.info("search_exact needle=%r", needle)

    for p in SEARCH_PATHS:
        path = Path(p)
        if not path.exists():
            continue

        txt = path.read_text(encoding="utf-8", errors="ignore")
        if needle in txt:
            hits.append(str(path))

    logger.info("search_exact_result needle=%r hits=%s", needle, hits)
    return ExactSearchResponse(found=bool(hits), files=hits)


# -------------------------------------------------
# QA / Answer endpoint
# -------------------------------------------------
agent = build_chain()  # async run(question) from chains.py


@app.post("/api/answer", response_model=AgentResponse)
async def api_answer(
    req: QueryRequest,
    auth: bool = Depends(require_auth),
):
    logger.info("user_question question=%r", req.question)

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
            answer_text = result.get("answer") or result.get("output") or ""
            confidence = result.get("confidence")
            logger.info(
                "answer_result confidence=%s answer_preview=%r",
                confidence,
                answer_text[:120],
            )
            return AgentResponse(
                answer=answer_text,
                confidence=confidence,
                error=result.get("error"),
                sources=result.get("sources") or [],
                chunk_count=result.get("chunk_count"),
            )

        # Fallback: treat anything else as plain text
        logger.info(
            "answer_result_plain confidence=0.5 answer_preview=%r",
            str(result)[:120],
        )
        return AgentResponse(answer=str(result), confidence=0.5, sources=[])

    except Exception as e:
        logger.exception("api_answer_failed: %s", e)
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
async def upload_file(
    file: UploadFile = File(...),
    auth: bool = Depends(require_auth),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file has no name")

    suffix = Path(file.filename).suffix or ".txt"
    tmp_name = f"ui_{uuid4().hex}{suffix}"
    tmp_path = UPLOAD_DIR / tmp_name

    contents = await file.read()
    tmp_path.write_bytes(contents)

    logger.info(
        "file_uploaded original=%r saved_as=%s size_bytes=%d",
        file.filename,
        tmp_path,
        len(contents),
    )

    return UploadResponse(
        server_path=str(tmp_path),
        original_filename=file.filename,
    )


# -------------------------------------------------
# Health check (left public so you can probe the server)
# -------------------------------------------------
@app.get("/api/health")
def health():
    return {"status": "ok"}


# -------------------------------------------------
# Logging + ingest status helpers
# -------------------------------------------------
@app.post("/api/log")
def log(
    ev: LogEvent,
    auth: bool = Depends(require_auth),
):
    # Log to file/logger (for live debugging)
    logger.info(
        "frontend_event event=%s status=%s file=%s ts=%s extra=%s",
        ev.event,
        ev.status,
        ev.file,
        ev.ts,
        ev.extra,
    )

    # Persist to SQLite
    try:
        insert_log_event(ev)
    except Exception as e:
        logger.exception("insert_log_event_failed: %s", e)

    return {
        "ok": True,
        "received": ev.model_dump(),
        "server_ts": datetime.utcnow().isoformat() + "Z",
    }

@app.post("/api/ingest_done")
def ingest_done(
    auth: bool = Depends(require_auth),
):
    global last_ingest_time
    last_ingest_time = datetime.utcnow()
    logger.info("ingest_done ts=%s", last_ingest_time.isoformat() + "Z")
    return {
        "status": "ok",
        "message": "Ingest completed",
        "ts": last_ingest_time,
    }


@app.get("/api/ingest_status")
def ingest_status(
    auth: bool = Depends(require_auth),
):
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
@app.get("/api/source_file")
def source_file(path: str, auth: bool = Depends(require_auth)):
    """
    Serve a cited source file so the UI can open/download it.
    We restrict to files under data/docs for safety.
    """
    requested = Path(path).resolve()
    base_dir = Path("data").resolve()

    # simple safety check: only allow paths under ./data
    if not str(requested).startswith(str(base_dir)):
        raise HTTPException(status_code=400, detail="Access denied")

    if not requested.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(requested)