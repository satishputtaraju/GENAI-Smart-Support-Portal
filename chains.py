# app/chains.py
import logging
from typing import List, Dict, Any, Optional
import os
import re
from pathlib import Path

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.retriever import load_vectorstore, TOP_K

# -------------------------------------------------------------------
# Global config
# -------------------------------------------------------------------

logger = logging.getLogger("support_agent")

LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1:latest")

DOCS_DIR = Path("data/docs")
UPLOAD_DIR = Path("data/uploads")

# Raw files we want to support for exact sentence lookup
RAW_FILES: List[Path] = [
    DOCS_DIR / "incident_utf8.txt",
    DOCS_DIR / "dialogueText.csv",
]

# Fallback incident file used for snippet search
INCIDENT_FILE = DOCS_DIR / "incident_utf8.txt"


# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------

def _exact_sentence_from_question(question: str) -> Optional[str]:
    """
    If the user asks something like:
      Can you check if the following sentence is available "...." ?
    extract the text between quotes and return it.
    """
    m = re.search(r'"([^"]+)"', question)
    if not m:
        return None

    sentence = m.group(1).strip()
    if len(sentence) < 5:
        # ignore very short “sentences”
        return None
    return sentence


def _search_sentence_in_files(sentence: str) -> List[str]:
    """
    Return list of file paths (as strings) where the exact sentence occurs.
    Case-sensitive substring search across RAW_FILES.
    """
    matches: List[str] = []
    for path in RAW_FILES:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if sentence in text:
            matches.append(str(path))
    return matches


def _read_incident_snippet(incident_id: str, radius_chars: int = 600) -> Optional[str]:
    """
    Fallback: open INCIDENT_FILE and extract a snippet
    around the given incident ID. Returns None if not found.
    """
    if not INCIDENT_FILE.exists():
        return None

    text = INCIDENT_FILE.read_text(encoding="utf-8", errors="ignore")
    up = text.upper()
    idx = up.find(incident_id.upper())
    if idx == -1:
        return None

    start = max(0, idx - radius_chars)
    end = min(len(text), idx + radius_chars)
    snippet = text[start:end]

    # Trim to full lines to make it nicer
    first_nl = snippet.find("\n")
    last_nl = snippet.rfind("\n")
    if first_nl != -1 and last_nl != -1 and last_nl > first_nl:
        snippet = snippet[first_nl + 1:last_nl]

    return snippet.strip()


def _extract_filename_mention(question: str) -> Optional[str]:
    """
    Try to detect a file name mentioned in the question, e.g.
      "summarize the content from the requirements.txt file"
    or "what's in dialogueText.csv?"
    """
    m = re.search(
        r'([\w\-.]+\.(txt|csv|md|log|json))',
        question,
        flags=re.IGNORECASE,
    )
    if not m:
        return None
    return m.group(1)


def _resolve_filename_to_path(fname: str) -> Optional[Path]:
    """
    Given a file name like 'requirements.txt', try to find it
    under known directories.
    """
    candidates = [
        DOCS_DIR / fname,
        UPLOAD_DIR / fname,
        Path(fname),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _dedupe_sources(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove duplicate (source,page) entries while preserving order.
    """
    seen = set()
    out: List[Dict[str, Any]] = []
    for s in sources:
        key = (s.get("source"), s.get("page"))
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


# -------------------------------------------------------------------
# Build the main chain
# -------------------------------------------------------------------

def build_chain(top_k: int = TOP_K):
    """
    Main assistant logic:

      0) If the question contains a quoted sentence, run an exact
         literal search across RAW_FILES.

      1) If the question mentions a specific file name (e.g. requirements.txt),
         prefer content from that file (vector search filtered by source);
         if that fails, fall back to reading the raw file and summarizing.

      2) If the question contains an incident ID like INC0640580,
         try vector search AND (if needed) raw text search in
         incident_utf8.txt.

      3) Otherwise, do normal semantic RAG over the vector store.
    """

    vectordb = load_vectorstore()
    retriever = vectordb.as_retriever(search_kwargs={"k": top_k})

    llm = ChatOllama(model=LLM_MODEL, temperature=0.2)

    # Generic RAG prompt
    generic_prompt = ChatPromptTemplate.from_template(
        "You are a helpful support assistant. Use ONLY the information in the "
        "following context (which may come from multiple support documents) "
        "to answer the question.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n"
        "Answer:"
    )

    # Incident summary prompt
    id_prompt = ChatPromptTemplate.from_template(
        "You are a helpful support assistant. The context below contains "
        "lines from an incident table.\n\n"
        "Focus on incident ID {incident_id} and write a clear summary using "
        "ONLY the information in the context. Include:\n"
        "- Brief description or short description\n"
        "- Opened date/time\n"
        "- Priority / impact / urgency (if present)\n"
        "- Assignment group or owner (if present)\n"
        "- Current status and any resolution/closure details (if present)\n\n"
        "Context:\n{context}\n\n"
        "Summary of incident {incident_id}:"
    )

    # File-focused prompt (summaries and file-scoped Q&A)
    file_prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant. The user is asking about the file "
        "{filename}. Use ONLY the following excerpts from that file to "
        "answer the question or provide a concise summary.\n\n"
        "File: {filename}\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n"
        "Answer:"
    )

    async def run(question: str) -> Dict[str, Any]:
        logger.info("chain_run_start question=%r", question)

        # ---------------------------------------------------
        # 0) Exact literal sentence search (raw files)
        # ---------------------------------------------------
        sentence = _exact_sentence_from_question(question)
        if sentence:
            logger.info("chain_branch=exact_sentence sentence=%r", sentence)
            files_with_sentence = _search_sentence_in_files(sentence)
            if files_with_sentence:
                sources = [{"source": f, "page": None} for f in files_with_sentence]
                sources = _dedupe_sources(sources)
                answer_text = (
                    f'Yes. I found the exact sentence "{sentence}" '
                    f"in the following file(s): {', '.join(files_with_sentence)}."
                )
                logger.info(
                    "exact_sentence_found files=%s answer_preview=%r",
                    files_with_sentence,
                    answer_text[:120],
                )
                return {
                    "answer": answer_text,
                    "confidence": 0.99,
                    "error": None,
                    "sources": sources,
                }
            else:
                logger.info("exact_sentence_not_found sentence=%r", sentence)
            # fall through to other logic

        # ---------------------------------------------------
        # 1) Question refers to a specific file name
        # ---------------------------------------------------
        fname = _extract_filename_mention(question)
        if fname:
            logger.info("chain_branch=file filename_mentioned=%s", fname)
            path = _resolve_filename_to_path(fname)
            if path:
                logger.info("file_resolved path=%s", path)

                # Prefer chunks that come specifically from this file
                try:
                    docs = vectordb.similarity_search(
                        question,
                        k=top_k * 4,
                        filter={"source": str(path)},
                    )
                    logger.info(
                        "file_branch_vector_search filter_docs=%d", len(docs)
                    )
                except TypeError:
                    # Older langchain_chroma may use "where" instead of "filter"
                    docs = vectordb.similarity_search(
                        question,
                        k=top_k * 4,
                        where={"source": str(path)},
                    )
                    logger.info(
                        "file_branch_vector_search_where docs=%d", len(docs)
                    )
                except Exception as e:
                    logger.exception(
                        "file_branch_vector_search_error path=%s error=%s", path, e
                    )
                    docs = []

                # Final fallback: unfiltered search + manual filter
                if not docs:
                    try:
                        all_docs = vectordb.similarity_search(
                            question, k=top_k * 4
                        )
                        docs = [
                            d
                            for d in all_docs
                            if d.metadata.get("source") == str(path)
                        ]
                        logger.info(
                            "file_branch_vector_search_manual_filter docs=%d",
                            len(docs),
                        )
                    except Exception as e:
                        logger.exception(
                            "file_branch_vector_search_manual_filter_error: %s", e
                        )
                        docs = []

                context_chunks: List[str] = []
                sources: List[Dict[str, Any]] = []

                if docs:
                    context_chunks = [d.page_content for d in docs]
                    sources = [
                        {
                            "source": d.metadata.get("source"),
                            "page": d.metadata.get("page"),
                        }
                        for d in docs
                    ]
                    logger.info(
                        "file_branch_docs_used count=%d unique_sources=%d",
                        len(docs),
                        len(_dedupe_sources(sources)),
                    )
                else:
                    # Fallback: use raw file contents
                    try:
                        text = path.read_text(encoding="utf-8", errors="ignore")
                        context_chunks = [text]
                        sources = [{"source": str(path), "page": None}]
                        logger.info(
                            "file_branch_fallback_raw_file path=%s length=%d",
                            path,
                            len(text),
                        )
                    except Exception as e:
                        logger.exception(
                            "file_branch_raw_read_failed path=%s error=%s", path, e
                        )
                        return {
                            "answer": "",
                            "confidence": 0.0,
                            "error": f"Could not read file {path}: {e}",
                            "sources": [],
                        }

                context = "\n\n".join(context_chunks)
                chain = file_prompt | llm | StrOutputParser()
                try:
                    answer_text: str = await chain.ainvoke(
                        {"context": context, "question": question, "filename": fname}
                    )
                    logger.info(
                        "file_branch_answer_done filename=%s answer_preview=%r",
                        fname,
                        answer_text[:120],
                    )
                except Exception as e:
                    logger.exception("file_branch_llm_error filename=%s error=%s", fname, e)
                    return {
                        "answer": "",
                        "confidence": 0.0,
                        "error": str(e),
                        "sources": _dedupe_sources(sources),
                    }

                return {
                    "answer": answer_text,
                    "confidence": 0.9,
                    "error": None,
                    "sources": _dedupe_sources(sources),
                }
            else:
                logger.info("file_not_found_for_name fname=%s", fname)

        # ---------------------------------------------------
        # 2) Detect an incident ID like INC0640580
        # ---------------------------------------------------
        m = re.search(r"(INC\d+)", question.upper())
        if m:
            incident_id = m.group(1)
            logger.info("chain_branch=incident incident_id=%s", incident_id)

            # --- A) Try vector search by ID across all ingested docs ---
            try:
                docs = vectordb.similarity_search(incident_id, k=top_k * 3)
                logger.info(
                    "incident_vector_search docs=%d", len(docs)
                )
            except Exception as e:
                logger.exception(
                    "incident_vector_search_error incident_id=%s error=%s",
                    incident_id,
                    e,
                )
                docs = []

            hit_docs = [
                d
                for d in docs
                if incident_id in (d.page_content or "").upper()
            ]

            context_chunks: List[str] = []
            sources: List[Dict[str, Any]] = []

            if hit_docs:
                for d in hit_docs:
                    context_chunks.append(d.page_content)
                    sources.append(
                        {
                            "source": d.metadata.get("source"),
                            "page": d.metadata.get("page"),
                        }
                    )
                logger.info(
                    "incident_hit_docs count=%d unique_sources=%d",
                    len(hit_docs),
                    len(_dedupe_sources(sources)),
                )

            # --- B) Fallback: raw text search only in INCIDENT_FILE ---
            if not context_chunks:
                snippet = _read_incident_snippet(incident_id)
                if snippet:
                    context_chunks.append(snippet)
                    sources.append({"source": str(INCIDENT_FILE), "page": None})
                    logger.info(
                        "incident_fallback_raw_snippet incident_id=%s snippet_len=%d",
                        incident_id,
                        len(snippet),
                    )

            if not context_chunks:
                logger.info(
                    "incident_not_found incident_id=%s", incident_id
                )
                return {
                    "answer": (
                        f'I couldn’t find incident ID "{incident_id}" '
                        f"in the provided records."
                    ),
                    "confidence": 0.7,
                    "error": None,
                    "sources": [],
                }

            context = "\n\n".join(context_chunks)
            chain = id_prompt | llm | StrOutputParser()
            try:
                answer_text: str = await chain.ainvoke(
                    {"context": context, "incident_id": incident_id}
                )
                logger.info(
                    "incident_answer_done incident_id=%s answer_preview=%r",
                    incident_id,
                    answer_text[:120],
                )
            except Exception as e:
                logger.exception(
                    "incident_llm_error incident_id=%s error=%s", incident_id, e
                )
                return {
                    "answer": "",
                    "confidence": 0.0,
                    "error": str(e),
                    "sources": _dedupe_sources(sources),
                }

            return {
                "answer": answer_text,
                "confidence": 0.9,
                "error": None,
                "sources": _dedupe_sources(sources),
            }

        # ---------------------------------------------------
        # 3) No incident ID / file name → normal semantic RAG
        # ---------------------------------------------------
        logger.info("chain_branch=generic")
        try:
            docs = retriever.get_relevant_documents(question)
            logger.info("generic_retriever_docs count=%d", len(docs))
        except Exception as e:
            logger.exception("generic_retriever_error: %s", e)
            docs = []

        # Fallback: if retriever returns nothing, try a direct similarity search
        if not docs:
            logger.info(
                "generic_retriever_empty -> fallback_similarity_search"
            )
            try:
                docs = vectordb.similarity_search(question, k=top_k * 4)
                logger.info(
                    "generic_fallback_similarity_docs count=%d", len(docs)
                )
            except Exception as e:
                logger.exception(
                    "generic_fallback_similarity_error: %s", e
                )
                docs = []

        if not docs:
            logger.info("generic_no_docs_found -> returning_idk")
            return {
                "answer": (
                    "I don't know. I couldn't find any relevant information in the "
                    "available documents."
                ),
                "confidence": 0.3,
                "error": None,
                "sources": [],
            }

        context = "\n\n".join(d.page_content for d in docs)
        chain = generic_prompt | llm | StrOutputParser()
        try:
            answer_text: str = await chain.ainvoke(
                {"context": context, "question": question}
            )
            logger.info(
                "generic_answer_done docs_used=%d answer_preview=%r",
                len(docs),
                answer_text[:120],
            )
        except Exception as e:
            logger.exception("generic_llm_error: %s", e)
            return {
                "answer": "",
                "confidence": 0.0,
                "error": str(e),
                "sources": [],
            }

        sources: List[Dict[str, Any]] = [
            {
                "source": d.metadata.get("source"),
                "page": d.metadata.get("page"),
            }
            for d in docs
        ]

        return {
            "answer": answer_text,
            "confidence": 0.9,
            "error": None,
            "sources": _dedupe_sources(sources),
        }

    # We return the async function
    return run
