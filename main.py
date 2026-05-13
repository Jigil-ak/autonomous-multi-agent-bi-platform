"""
Autonomous Multi-Agent Business Intelligence & Execution Platform
==================================================================
FastAPI application entry point.

Start with:
    python main.py
or:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import router
from config import PROJECT_NAME, PROJECT_VERSION
from utils.logger import get_logger

logger = get_logger("main")


# ---------------------------------------------------------------------------
# Lifespan events
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup
    logger.info("=" * 60)
    logger.info(f"Starting {PROJECT_NAME} v{PROJECT_VERSION}")
    logger.info("=" * 60)

    # Ensure directories exist
    for d in ["logs", "reports", "chroma_db"]:
        Path(d).mkdir(exist_ok=True)

    # Pre-warm ChromaDB singleton (avoids cold-start delay on first request)
    try:
        logger.info("Pre-warming ChromaDB memory store...")
        from memory.chroma_store import get_memory_store
        store = get_memory_store()
        logger.info(f"ChromaDB ready. Documents in store: {store.document_count()}")
    except Exception as exc:
        logger.warning(f"ChromaDB pre-warm failed (non-fatal): {exc}")

    logger.info("FastAPI server ready. Multi-agent platform is live.")
    logger.info("API docs available at: http://localhost:8000/docs")

    yield

    # Shutdown
    logger.info(f"Shutting down {PROJECT_NAME}...")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title=PROJECT_NAME,
    version=PROJECT_VERSION,
    description=(
        "A multi-agent AI platform that coordinates Research, Strategy, Planning, "
        "Critic, QA, and Memory agents using Gemini models to generate comprehensive "
        "business intelligence and execution plans."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow Streamlit (localhost:8501) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routes
app.include_router(router, prefix="")


# ---------------------------------------------------------------------------
# Health & root endpoints
# ---------------------------------------------------------------------------
@app.get("/", tags=["health"])
async def root():
    """Platform root — basic health check."""
    return {
        "platform": PROJECT_NAME,
        "version": PROJECT_VERSION,
        "status": "running",
        "timestamp": time.time(),
        "endpoints": {
            "analyze":  "POST /analyze",
            "status":   "GET  /status/{task_id}",
            "logs":     "GET  /logs/{task_id}",
            "report":   "GET  /report/{task_id}",
            "tasks":    "GET  /tasks",
            "docs":     "GET  /docs",
        },
    }


@app.get("/health", tags=["health"])
async def health():
    """Detailed health check."""
    checks: dict[str, str] = {}

    # ChromaDB
    try:
        from memory.chroma_store import get_memory_store
        store = get_memory_store()
        checks["chromadb"] = f"OK ({store.document_count()} docs)"
    except Exception as exc:
        checks["chromadb"] = f"ERROR: {exc}"

    # API key
    from config import GOOGLE_API_KEY
    checks["google_api_key"] = "SET" if GOOGLE_API_KEY else "MISSING"

    # Reports dir
    from api.routes import REPORTS_DIR
    report_count = len(list(REPORTS_DIR.glob("*.json")))
    checks["reports_on_disk"] = str(report_count)

    all_ok = all("ERROR" not in v for v in checks.values())

    return JSONResponse(
        content={"status": "healthy" if all_ok else "degraded", "checks": checks},
        status_code=200 if all_ok else 207,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disable reload to avoid ChromaDB re-init issues
        log_level="info",
    )
