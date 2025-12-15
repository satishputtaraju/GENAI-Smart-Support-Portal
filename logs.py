# app/logs.py
from fastapi import APIRouter
from pydantic import BaseModel
from datetime import datetime

router = APIRouter(prefix="/api", tags=["logs"])

class LogEvent(BaseModel):
    event: str
    status: str = "ok"
    file: str | None = None
    ts: str | None = None
    meta: dict | None = None

@router.post("/log")
def log_event(entry: LogEvent):
    # TODO: optionally write to SQLite (support.db)
    return {
        "ok": True,
        "received_at": datetime.utcnow().isoformat() + "Z",
        "entry": entry.model_dump(),
    }