"""
Structured Logging Utility
===========================
Provides consistent, structured logging across all agents and modules.

Features:
- Timestamped log entries with agent names
- File + console output handlers
- Execution tracing with duration and status
- Token usage estimation helper
"""

import logging
import time
from pathlib import Path
from typing import Any

from config import LOG_LEVEL, LOG_DIR


# ---------------------------------------------------------------------------
# Custom Formatter
# ---------------------------------------------------------------------------
class AgentFormatter(logging.Formatter):
    """Produces structured log lines with timestamps and context."""

    FORMAT = (
        "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
    )
    DATE_FMT = "%Y-%m-%d %H:%M:%S"

    def __init__(self) -> None:
        super().__init__(fmt=self.FORMAT, datefmt=self.DATE_FMT)


# ---------------------------------------------------------------------------
# Logger Factory
# ---------------------------------------------------------------------------
_loggers: dict[str, logging.Logger] = {}


def get_logger(agent_name: str) -> logging.Logger:
    """
    Return a configured logger for the given agent / module name.

    Loggers are cached so the same name always returns the same instance
    with handlers attached exactly once.
    """
    if agent_name in _loggers:
        return _loggers[agent_name]

    logger = logging.getLogger(agent_name)
    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    # Prevent duplicate handlers when module is reloaded
    if not logger.handlers:
        formatter = AgentFormatter()

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler
        log_file = LOG_DIR / "platform.log"
        file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Don't propagate to root logger
    logger.propagate = False

    _loggers[agent_name] = logger
    return logger


# ---------------------------------------------------------------------------
# Execution Tracing
# ---------------------------------------------------------------------------
def log_execution(
    agent_name: str,
    action: str,
    start_time: float,
    end_time: float,
    status: str = "success",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Log an execution trace for an agent action and return the trace record.

    Parameters
    ----------
    agent_name : str
        Name of the agent performing the action.
    action : str
        Description of the action (e.g. 'research_competitors').
    start_time : float
        ``time.time()`` when the action started.
    end_time : float
        ``time.time()`` when the action completed.
    status : str
        Outcome — ``'success'``, ``'failed'``, ``'fallback'``.
    metadata : dict, optional
        Arbitrary key-value pairs (token estimates, model used, etc.).

    Returns
    -------
    dict
        The structured trace record.
    """
    logger = get_logger(agent_name)
    duration_ms = round((end_time - start_time) * 1000, 2)

    trace = {
        "agent": agent_name,
        "action": action,
        "status": status,
        "duration_ms": duration_ms,
        "start_time": start_time,
        "end_time": end_time,
        "metadata": metadata or {},
    }

    msg = (
        f"action={action} | status={status} | "
        f"duration={duration_ms}ms"
    )
    if metadata:
        extras = " | ".join(f"{k}={v}" for k, v in metadata.items())
        msg += f" | {extras}"

    if status == "success":
        logger.info(msg)
    elif status == "fallback":
        logger.warning(msg)
    else:
        logger.error(msg)

    return trace


# ---------------------------------------------------------------------------
# Token Estimation
# ---------------------------------------------------------------------------
def estimate_tokens(text: str) -> int:
    """
    Rough token estimate (≈ 4 characters per token for English text).
    This is intentionally simple to avoid adding a tokenizer dependency.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Timer Context Manager
# ---------------------------------------------------------------------------
class ExecutionTimer:
    """
    Context manager that tracks wall-clock time.

    Usage::

        with ExecutionTimer() as t:
            do_work()
        print(t.duration_ms)
    """

    def __init__(self) -> None:
        self.start: float = 0.0
        self.end: float = 0.0
        self.duration_ms: float = 0.0

    def __enter__(self) -> "ExecutionTimer":
        self.start = time.time()
        return self

    def __exit__(self, *_: Any) -> None:
        self.end = time.time()
        self.duration_ms = round((self.end - self.start) * 1000, 2)
