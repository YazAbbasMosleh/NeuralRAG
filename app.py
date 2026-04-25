"""
FastAPI backend for RAG Chatbot
Run with: uvicorn api:app --reload --port 8000

Place this file at the root of your project (same level as utils/ and src/).
Serve the frontend by putting static/index.html next to it.
"""

import os
import logging
import shutil
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from utils.config import Config
from src.document_processor import DocumentProcessor
from src.embeddings import Embeddings
from src.vector_store import VectorStore
from src.llm import LLM
from src.rag_pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Chatbot API",
    description="Local RAG pipeline with GGUF embeddings + Ollama LLMs",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static frontend files
app.mount("/static", StaticFiles(directory="static"), name="static")

# ---------------------------------------------------------------------------
# Global runtime state  (single-user / demo)
# ---------------------------------------------------------------------------
state: dict = {
    "config": None,
    "embedding_model": None,
    "vector_store": None,
    "llm": None,
    "rag_pipeline": None,
    "loaded_pdf": None,
    "status": "idle",   # idle | loading | ready | error
    "error": None,
}

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class InferenceRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    k: int = Field(4, ge=1, le=20, description="Retrieved chunks count")


class StatusResponse(BaseModel):
    status: str
    loaded_pdf: Optional[str] = None
    llm_model: Optional[str] = None
    embedding_model: Optional[str] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=FileResponse)
async def serve_frontend():
    return FileResponse("static/index.html")


@app.get("/status", response_model=StatusResponse)
async def get_status():
    cfg = state.get("config")
    return StatusResponse(
        status=state["status"],
        loaded_pdf=state.get("loaded_pdf"),
        llm_model=getattr(cfg, "llm_model", None),
        embedding_model=getattr(cfg, "embedding_model_path", None),
        error=state.get("error"),
    )


@app.post("/load")
async def load_model_and_pdf(
    pdf_file: UploadFile = File(..., description="PDF document to ingest"),
    llm_model: str = Form("qwen2.5-coder:0.5b"),
    embedding_model_path: str = Form("embedding_models/nomic-embed-text-v2.gguf"),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200),
    chunking_method: str = Form("recursive_character"),
    temperature: float = Form(0.7),
    top_p: float = Form(0.9),
    n_ctx: int = Form(2048),
    max_tokens: int = Form(2048),
    vector_store_path: str = Form("vector_store"),
):
    """Upload a PDF and initialise the full RAG stack with the given settings."""

    if not pdf_file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are accepted.")

    state["status"] = "loading"
    state["error"] = None

    try:
        # ---- Save PDF ----
        pdf_path = UPLOAD_DIR / pdf_file.filename
        with open(pdf_path, "wb") as f:
            shutil.copyfileobj(pdf_file.file, f)

        # ---- Lightweight config object (no YAML needed) ----
        class _Cfg:
            pass

        cfg = _Cfg()
        cfg.llm_model = llm_model
        cfg.embedding_model_path = embedding_model_path
        cfg.pdf_path = str(pdf_path)
        cfg.vector_store_path = vector_store_path
        cfg.chunk_size = chunk_size
        cfg.chunk_overlap = chunk_overlap
        cfg.temperature = temperature
        cfg.top_p = top_p
        cfg.n_ctx = n_ctx
        cfg.max_tokens = max_tokens

        if not embedding_model_path.endswith(".gguf"):
            raise ValueError("Embedding model path must end with .gguf")
        if not os.path.exists(embedding_model_path):
            raise FileNotFoundError(f"Embedding model not found: {embedding_model_path}")

        # ---- Process PDF ----
        processor = DocumentProcessor(
            chunking_method=chunking_method,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        docs = processor.load_pdf(str(pdf_path))
        chunks = processor.split_document(docs)

        # ---- Embeddings ----
        embedding_model = Embeddings(config=cfg, model_path=embedding_model_path)

        # ---- Vector store ----
        vs = VectorStore(embedding_model=embedding_model, persist_path=vector_store_path)
        vs.build(chunks)
        vs.save()

        # ---- LLM ----
        llm_obj = LLM(config=cfg, model_name=llm_model)
        llm = llm_obj.get()

        # ---- RAG pipeline ----
        rag = RAGPipeline(llm=llm, vector_store=vs)

        # ---- Persist ----
        state.update({
            "config": cfg,
            "embedding_model": embedding_model,
            "vector_store": vs,
            "llm": llm,
            "rag_pipeline": rag,
            "loaded_pdf": pdf_file.filename,
            "status": "ready",
        })

        return {
            "message": "RAG system loaded successfully.",
            "pdf": pdf_file.filename,
            "chunks": len(chunks),
            "llm_model": llm_model,
            "embedding_model": embedding_model_path,
        }

    except Exception as exc:
        state["status"] = "error"
        state["error"] = str(exc)
        logger.exception("Failed to load RAG system")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/infer")
async def infer(req: InferenceRequest):
    """Run a query through the RAG pipeline."""
    if state["status"] != "ready":
        raise HTTPException(
            status_code=400,
            detail=f"System not ready (status={state['status']}). Call /load first.",
        )
    try:
        rag: RAGPipeline = state["rag_pipeline"]

        # Override retrieve_context to respect per-request k
        def _retrieve(query: str):
            docs = state["vector_store"].similarity_search(query, k=req.k)
            return "\n\n".join([d.page_content for d in docs])

        rag.retrieve_context = _retrieve
        answer = rag.run(req.query)
        return {"query": req.query, "answer": answer}

    except Exception as exc:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.delete("/reset")
async def reset():
    """Tear down the loaded model and pipeline."""
    for key in ("config", "embedding_model", "vector_store", "llm", "rag_pipeline",
                "loaded_pdf", "error"):
        state[key] = None
    state["status"] = "idle"
    return {"message": "System reset."}
