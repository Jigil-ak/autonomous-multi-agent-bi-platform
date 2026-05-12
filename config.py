"""
Centralized Configuration Module
=================================
Loads environment variables and exposes typed settings used across the platform.
All configuration is centralized here to avoid scattered env lookups.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env file from project root
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(_PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Google Gemini API
# ---------------------------------------------------------------------------
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")

# ---------------------------------------------------------------------------
# ChromaDB
# ---------------------------------------------------------------------------
CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", "./chroma_db")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_DIR: Path = _PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Agent Safety Limits
# ---------------------------------------------------------------------------
MAX_AGENT_ITERATIONS: int = int(os.getenv("MAX_AGENT_ITERATIONS", "5"))
MAX_WORKFLOW_TOKENS: int = int(os.getenv("MAX_WORKFLOW_TOKENS", "100000"))
MAX_CONCURRENT_WORKFLOWS: int = int(os.getenv("MAX_CONCURRENT_WORKFLOWS", "3"))

# ---------------------------------------------------------------------------
# Retry Configuration
# ---------------------------------------------------------------------------
RETRY_DELAYS: list[int] = [2, 4, 8]  # seconds between retries
MAX_RETRIES: int = 3

# ---------------------------------------------------------------------------
# Context Window Management
# ---------------------------------------------------------------------------
MAX_TOKENS_CONTEXT: int = 10000  # Summarize history if exceeded

# ---------------------------------------------------------------------------
# Model Routing
# ---------------------------------------------------------------------------
# Maps agent roles to their primary Gemini model.
MODEL_ROUTER: dict[str, str] = {
    "orchestrator": "gemini-2.5-flash",
    "research":     "gemini-2.5-flash",
    "strategy":     "gemini-2.5-pro",
    "planning":     "gemini-2.5-flash",
    "critic":       "gemini-2.5-pro",
    "qa":           "gemini-2.5-flash-lite",
    "memory":       "gemini-2.5-flash-lite",
}

# Fallback chain: if primary model fails, try the next in the list.
MODEL_FALLBACKS: dict[str, list[str]] = {
    "gemini-2.5-pro":        ["gemini-2.5-flash", "gemini-2.5-flash-lite"],
    "gemini-2.5-flash":      ["gemini-2.5-flash-lite"],
    "gemini-2.5-flash-lite": [],  # No further fallback; use mock response
}

# ---------------------------------------------------------------------------
# DuckDuckGo Search
# ---------------------------------------------------------------------------
SEARCH_DELAY_SECONDS: float = 2.0  # Delay between search queries

# ---------------------------------------------------------------------------
# Project Metadata
# ---------------------------------------------------------------------------
PROJECT_NAME: str = "Autonomous Multi-Agent BI Platform"
PROJECT_VERSION: str = "1.0.0"
