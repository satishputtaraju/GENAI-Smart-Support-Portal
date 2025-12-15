# app/ingest.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path
import shutil
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader,
    CSVLoader,
    PyPDFLoader,
)
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


router = APIRouter(prefix="/api", tags=["ingest"])

#PERSIST_DIR = Path("data/chroma")
#PERSIST_DIR.mkdir(parents=True, exist_ok=True)

PERSIST_DIR = Path(os.getenv("CHROMA_DIR", "data/chroma"))
PERSIST_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Request Model ----------
class IngestRequest(BaseModel):
    paths: List[str]
    reset_collection: bool = False
    chunk_size: int = 1000
    chunk_overlap: int = 150


# ---------- File Loader ----------
def load_file(path: Path):
    suffix = path.suffix.lower()

#    if suffix in [".txt", ".md"]:
#        return TextLoader(str(path), encoding="utf8").load()

# Treat CSV as plain text for now (more robust)
    if suffix in [".txt", ".md", ".csv"]:
        return TextLoader(str(path), encoding="utf8").load()

#    if suffix == ".csv":
#        return CSVLoader(str(path)).load()

    if suffix == ".pdf":
        return PyPDFLoader(str(path)).load()

    return []


# ---------- Ingest Endpoint ----------
@router.post("/ingest")
def ingest(req: IngestRequest):

    try:
        # reset collection if requested
        if req.reset_collection and PERSIST_DIR.exists():
            shutil.rmtree(PERSIST_DIR)
            PERSIST_DIR.mkdir(parents=True, exist_ok=True)

        # load docs
        docs = []
        for p in req.paths:
            fp = Path(p)
            if not fp.exists():
                raise HTTPException(status_code=400, detail=f"File not found: {fp}")

            if fp.is_file():
                docs.extend(load_file(fp))
            else:
                raise HTTPException(status_code=400, detail=f"Not a file: {fp}")

        if not docs:
            raise HTTPException(
                status_code=500, detail="Unable to read any documents from paths."
            )

        # split docs
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=req.chunk_size,
            chunk_overlap=req.chunk_overlap,
        )
        chunks = splitter.split_documents(docs)

        if not chunks:
            raise HTTPException(status_code=500, detail="No chunks produced.")

        # create embeddings + chroma
        try:
        #    emb = OllamaEmbeddings(model="nomic-embed-text")
            emb = OllamaEmbeddings(model="nomic-embed-text-v2-moe")    
            db = Chroma(
                collection_name="support_docs",
                persist_directory=str(PERSIST_DIR),
                embedding_function=emb,
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to initialize Chroma/Ollama: {e}",
            )

        # add chunks in safe batch sizes
        total_added = 0
        BATCH = 500

        for i in range(0, len(chunks), BATCH):
            batch = chunks[i:i+BATCH]
            db.add_documents(batch)
            total_added += len(batch)

        # final count
        try:
            count_after = db._collection.count()
        except Exception:
            count_after = None

        return {
            "status": "ok",
            "files_ingested": req.paths,
            "chunks_created": len(chunks),
            "chunks_added": total_added,
            "collection_count_after": count_after,
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
