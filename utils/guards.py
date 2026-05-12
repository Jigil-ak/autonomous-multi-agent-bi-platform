"""
Security Guards & Cost Tracking
=================================
Lightweight protections for the multi-agent platform.

Features:
- Prompt injection filtering (regex-based)
- Agent iteration limit enforcement
- Cost tracking with token estimation and request counting
- Workflow budget enforcement
"""

import re
from typing import Any

from config import MAX_AGENT_ITERATIONS, MAX_WORKFLOW_TOKENS
from utils.logger import get_logger, estimate_tokens

logger = get_logger("guards")


# ---------------------------------------------------------------------------
# Prompt Injection Protection
# ---------------------------------------------------------------------------
# Patterns that indicate prompt injection attempts
_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"ignore\s+(all\s+)?above\s+instructions", re.IGNORECASE),
    re.compile(r"system\s+override", re.IGNORECASE),
    re.compile(r"prompt\s+leak", re.IGNORECASE),
    re.compile(r"reveal\s+(your|the)\s+(system\s+)?prompt", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?prior", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(a|an)\s+", re.IGNORECASE),
    re.compile(r"act\s+as\s+if\s+you\s+have\s+no\s+restrictions", re.IGNORECASE),
    re.compile(r"bypass\s+(safety|content)\s+(filter|restriction)", re.IGNORECASE),
    re.compile(r"jailbreak", re.IGNORECASE),
    re.compile(r"DAN\s+mode", re.IGNORECASE),
]


def sanitize_input(text: str) -> str:
    """
    Scan user input for prompt injection patterns and remove them.

    Parameters
    ----------
    text : str
        Raw user input text.

    Returns
    -------
    str
        Sanitized text with injection patterns removed.
    """
    sanitized = text
    injections_found: list[str] = []

    for pattern in _INJECTION_PATTERNS:
        matches = pattern.findall(sanitized)
        if matches:
            injections_found.extend(matches if isinstance(matches[0], str) else [str(m) for m in matches])
            sanitized = pattern.sub("[FILTERED]", sanitized)

    if injections_found:
        logger.warning(
            f"Prompt injection detected and filtered: "
            f"{len(injections_found)} pattern(s) matched."
        )

    return sanitized.strip()


def is_input_safe(text: str) -> bool:
    """
    Check if user input contains prompt injection patterns.

    Returns True if no injection patterns are found.
    """
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            return False
    return True


# ---------------------------------------------------------------------------
# Iteration Limit Protection
# ---------------------------------------------------------------------------
def check_iteration_limit(
    current: int,
    max_iterations: int = MAX_AGENT_ITERATIONS,
) -> bool:
    """
    Check whether the current iteration count is within the allowed limit.

    Parameters
    ----------
    current : int
        Current iteration number (1-based).
    max_iterations : int
        Maximum allowed iterations.

    Returns
    -------
    bool
        True if more iterations are allowed, False if the limit is reached.
    """
    if current >= max_iterations:
        logger.warning(
            f"Iteration limit reached: {current}/{max_iterations}. "
            f"Stopping agent to prevent infinite loop."
        )
        return False
    return True


# ---------------------------------------------------------------------------
# Cost Tracker
# ---------------------------------------------------------------------------
class CostTracker:
    """
    Tracks estimated token usage and request counts per workflow.

    Usage::

        tracker = CostTracker(max_tokens=100000)
        tracker.add_usage(prompt, response)
        if tracker.can_proceed():
            # continue with next agent
    """

    def __init__(
        self,
        max_tokens: int = MAX_WORKFLOW_TOKENS,
        max_requests: int = 50,
    ) -> None:
        self.max_tokens = max_tokens
        self.max_requests = max_requests
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.request_count: int = 0
        self._usage_log: list[dict[str, Any]] = []

    @property
    def total_tokens(self) -> int:
        """Total estimated tokens (input + output)."""
        return self.total_input_tokens + self.total_output_tokens

    def add_usage(
        self,
        prompt: str,
        response: str,
        agent_name: str = "unknown",
    ) -> None:
        """Record token usage for a single LLM call."""
        input_tokens = estimate_tokens(prompt)
        output_tokens = estimate_tokens(response)

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.request_count += 1

        self._usage_log.append({
            "agent": agent_name,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cumulative_tokens": self.total_tokens,
            "request_number": self.request_count,
        })

        logger.debug(
            f"Cost update: agent={agent_name} | "
            f"+{input_tokens}in +{output_tokens}out | "
            f"total={self.total_tokens}/{self.max_tokens} tokens | "
            f"requests={self.request_count}/{self.max_requests}"
        )

    def can_proceed(self) -> bool:
        """
        Check if the workflow is within budget.

        Returns True if both token and request limits have not been exceeded.
        """
        if self.total_tokens >= self.max_tokens:
            logger.warning(
                f"Token budget exhausted: {self.total_tokens}/{self.max_tokens}. "
                f"Halting workflow."
            )
            return False

        if self.request_count >= self.max_requests:
            logger.warning(
                f"Request limit reached: {self.request_count}/{self.max_requests}. "
                f"Halting workflow."
            )
            return False

        return True

    def get_summary(self) -> dict[str, Any]:
        """Return a summary of cost tracking data."""
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "request_count": self.request_count,
            "max_tokens": self.max_tokens,
            "max_requests": self.max_requests,
            "budget_remaining_tokens": max(0, self.max_tokens - self.total_tokens),
            "budget_remaining_requests": max(0, self.max_requests - self.request_count),
            "usage_log": self._usage_log,
        }

    def reset(self) -> None:
        """Reset all tracking counters."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.request_count = 0
        self._usage_log.clear()
        logger.info("Cost tracker reset.")
