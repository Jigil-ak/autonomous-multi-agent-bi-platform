"""
Memory Agent
=============
Stores, retrieves, and compresses workflow context using ChromaDB.

Model: gemini-2.5-flash-lite (primary), gemini-2.5-flash (fallback)

Responsibilities:
- Store each agent's output in persistent vector memory
- Retrieve semantically relevant historical context on demand
- Summarize memory when it exceeds the token threshold (10k tokens)
- Expose a clean retrieval interface for other agents
"""

from __future__ import annotations

import time
from typing import Optional

from agents.schemas import (
    MemoryEntry, MemoryRetrievalResult, WorkflowState, AgentStatus,
)
from memory.chroma_store import get_memory_store
from utils.logger import get_logger, log_execution, estimate_tokens
from utils.model_router import call_model

logger = get_logger("memory_agent")

_SYSTEM_PROMPT = """You are a memory compression specialist.
Summarize the provided conversation history into a concise, information-dense summary.
Preserve: key facts, strategic decisions, competitor names, market signals, KPIs, and action items.
Discard: repetition, verbose explanations, formatting noise.
Return only the compressed summary as plain text.
"""


class MemoryAgent:
    """
    Memory Agent — manages persistent context across the workflow.
    """

    AGENT_NAME = "memory_agent"
    ROLE = "memory"

    def __init__(self) -> None:
        self._store = get_memory_store()

    # ------------------------------------------------------------------
    # Store
    # ------------------------------------------------------------------

    def store(
        self,
        task_id: str,
        agent_name: str,
        content: str,
        metadata: Optional[dict] = None,
    ) -> str:
        """
        Store agent output in vector memory.

        Returns the document ID or empty string on failure.
        """
        start_time = time.time()
        doc_id = self._store.store(
            task_id=task_id,
            agent_name=agent_name,
            content=content,
            metadata=metadata,
        )

        # Check if summarization is needed after storing
        if self._store.needs_summarization(task_id):
            logger.info(
                f"[{self.AGENT_NAME}] Context threshold reached for "
                f"task={task_id}. Triggering compression."
            )
            self.compress_context(task_id)

        log_execution(
            agent_name=self.AGENT_NAME,
            action="store",
            start_time=start_time,
            end_time=time.time(),
            status="success",
            metadata={"task_id": task_id, "source_agent": agent_name},
        )
        return doc_id

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def retrieve(
        self,
        task_id: str,
        query: str,
        top_k: int = 5,
        filter_agent: Optional[str] = None,
    ) -> MemoryRetrievalResult:
        """
        Semantic retrieval of relevant historical context.

        Parameters
        ----------
        task_id : str
            Workflow to query within.
        query : str
            Natural language query.
        top_k : int
            Number of results to return.
        filter_agent : str, optional
            Restrict results to a specific agent's outputs.

        Returns
        -------
        MemoryRetrievalResult
            Structured retrieval result with entries.
        """
        start_time = time.time()

        raw_results = self._store.retrieve(
            task_id=task_id,
            query=query,
            top_k=top_k,
            filter_agent=filter_agent,
        )

        entries = [
            MemoryEntry(
                task_id=task_id,
                agent_name=r["metadata"].get("agent_name", "unknown"),
                content=r["content"],
                metadata=r["metadata"],
                timestamp=float(r["metadata"].get("timestamp", time.time())),
                entry_id=r.get("entry_id"),
            )
            for r in raw_results
        ]

        log_execution(
            agent_name=self.AGENT_NAME,
            action="retrieve",
            start_time=start_time,
            end_time=time.time(),
            status="success",
            metadata={"task_id": task_id, "query": query[:60], "results": len(entries)},
        )

        return MemoryRetrievalResult(
            query=query,
            entries=entries,
            total_found=len(entries),
        )

    # ------------------------------------------------------------------
    # Compress
    # ------------------------------------------------------------------

    def compress_context(self, task_id: str) -> str:
        """
        Summarize conversation history if it exceeds the token threshold.

        Uses gemini-2.5-flash-lite for lightweight compression.
        Stores the resulting summary back into memory.

        Returns
        -------
        str
            The compressed summary text.
        """
        start_time = time.time()

        history = self._store.get_conversation_history(task_id, limit=50)
        if not history:
            return ""

        full_text = "\n\n".join(
            f"[{e['metadata'].get('agent_name', 'unknown')}]: {e['content']}"
            for e in history
        )

        total_tokens = estimate_tokens(full_text)
        logger.info(
            f"[{self.AGENT_NAME}] Compressing ~{total_tokens} tokens "
            f"for task={task_id}"
        )

        prompt = f"""Please compress the following workflow conversation history:

{full_text}

Create a dense, factual summary preserving all strategic decisions, 
competitor names, market data, KPIs, and action items.
"""

        try:
            summary = call_model(
                role=self.ROLE,
                prompt=prompt,
                system_prompt=_SYSTEM_PROMPT,
                temperature=0.1,
            )
        except Exception as exc:
            logger.error(f"[{self.AGENT_NAME}] Compression LLM call failed: {exc}")
            # Fallback: truncate oldest content
            summary = (
                "[COMPRESSED SUMMARY]\n" +
                full_text[-3000:]  # Keep last ~750 tokens
            )

        # Store the summary
        self._store.store_summary(task_id, summary)

        log_execution(
            agent_name=self.AGENT_NAME,
            action="compress_context",
            start_time=start_time,
            end_time=time.time(),
            status="success",
            metadata={
                "task_id": task_id,
                "original_tokens": total_tokens,
                "summary_tokens": estimate_tokens(summary),
            },
        )

        logger.info(
            f"[{self.AGENT_NAME}] Compression complete. "
            f"{total_tokens} → {estimate_tokens(summary)} tokens."
        )
        return summary

    # ------------------------------------------------------------------
    # Get formatted context for agent prompts
    # ------------------------------------------------------------------

    def get_context_for_prompt(
        self,
        task_id: str,
        query: str,
        top_k: int = 3,
    ) -> str:
        """
        Return a formatted string of relevant memory entries
        suitable for injection into an agent prompt.
        """
        result = self.retrieve(task_id, query, top_k=top_k)

        if not result.entries:
            return "No relevant historical context found."

        lines = ["=== Relevant Context from Memory ==="]
        for entry in result.entries:
            lines.append(
                f"\n[{entry.agent_name}]:\n{entry.content[:500]}"
            )
        lines.append("=== End of Context ===")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Run (used by orchestrator for explicit memory operations)
    # ------------------------------------------------------------------

    def run(
        self,
        task_id: str,
        agent_name: str,
        content: str,
        workflow_state: Optional[WorkflowState] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        """Store content and return document ID. Used by orchestrator."""
        if workflow_state:
            workflow_state.mark_agent_start(self.AGENT_NAME, "gemini-2.5-flash-lite")

        doc_id = self.store(task_id, agent_name, content, metadata)

        if workflow_state:
            workflow_state.mark_agent_done(self.AGENT_NAME, AgentStatus.SUCCESS)

        return doc_id


# ---------------------------------------------------------------------------
# Module-level singleton for cross-agent use
# ---------------------------------------------------------------------------
_memory_agent_instance: Optional[MemoryAgent] = None


def get_memory_agent() -> MemoryAgent:
    """Return the shared MemoryAgent singleton."""
    global _memory_agent_instance
    if _memory_agent_instance is None:
        _memory_agent_instance = MemoryAgent()
    return _memory_agent_instance
