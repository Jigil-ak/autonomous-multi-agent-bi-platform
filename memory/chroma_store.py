"""
ChromaDB Singleton Memory Store
=================================
Thread-safe, persistent vector memory shared across all agents.

Design decisions:
- Singleton pattern prevents duplicate client initialization and SQLite lock errors
- Uses sentence-transformers (all-MiniLM-L6-v2) for lightweight local embeddings
- Persistent storage at CHROMA_DB_PATH (configurable via .env)
- Thread lock on all write operations to prevent concurrent write conflicts
- Context window management: summarizes history if it exceeds MAX_TOKENS_CONTEXT

Usage:
    store = get_memory_store()         # always returns the same instance
    store.store("task-1", "research", "Competitor X dominates...")
    results = store.retrieve("task-1", "who are the competitors?")
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any, Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from config import CHROMA_DB_PATH, MAX_TOKENS_CONTEXT
from utils.logger import get_logger, estimate_tokens

logger = get_logger("chroma_store")

# ---------------------------------------------------------------------------
# Embedding model — loaded once globally (lightweight, CPU-only)
# ---------------------------------------------------------------------------
_embedding_model: Optional[SentenceTransformer] = None
_embedding_lock = threading.Lock()


def _get_embedding_model() -> SentenceTransformer:
    """Lazy-load the sentence-transformer model (loads only once)."""
    global _embedding_model
    if _embedding_model is None:
        with _embedding_lock:
            if _embedding_model is None:
                logger.info("Loading sentence-transformer model: all-MiniLM-L6-v2")
                _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
                logger.info("Embedding model loaded successfully.")
    return _embedding_model


# ---------------------------------------------------------------------------
# ChromaDB custom embedding function
# ---------------------------------------------------------------------------
class SentenceTransformerEmbedding(chromadb.EmbeddingFunction):
    """Wraps sentence-transformers to satisfy ChromaDB's embedding interface."""

    def __call__(self, input: list[str]) -> list[list[float]]:
        model = _get_embedding_model()
        embeddings = model.encode(input, convert_to_numpy=True)
        return embeddings.tolist()


# ---------------------------------------------------------------------------
# Singleton ChromaMemoryStore
# ---------------------------------------------------------------------------
_store_instance: Optional["ChromaMemoryStore"] = None
_store_lock = threading.Lock()


def get_memory_store() -> "ChromaMemoryStore":
    """Return the shared ChromaMemoryStore singleton."""
    global _store_instance
    if _store_instance is None:
        with _store_lock:
            if _store_instance is None:
                _store_instance = ChromaMemoryStore()
    return _store_instance


class ChromaMemoryStore:
    """
    Thread-safe persistent vector memory store.

    All write operations are protected by a lock to prevent SQLite
    concurrent write conflicts under FastAPI async workloads.
    """

    COLLECTION_NAME = "agent_memory"

    def __init__(self) -> None:
        self._write_lock = threading.Lock()
        self._embedding_fn = SentenceTransformerEmbedding()
        self._client = self._init_client()
        self._collection = self._init_collection()
        logger.info(
            f"ChromaMemoryStore initialized. "
            f"DB path: {CHROMA_DB_PATH} | "
            f"Collection: {self.COLLECTION_NAME}"
        )

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_client(self) -> chromadb.ClientAPI:
        """Create a persistent ChromaDB client."""
        try:
            client = chromadb.PersistentClient(
                path=CHROMA_DB_PATH,
                settings=Settings(anonymized_telemetry=False),
            )
            logger.info("ChromaDB persistent client created.")
            return client
        except Exception as exc:
            logger.error(f"Failed to create ChromaDB client: {exc}")
            raise

    def _init_collection(self) -> chromadb.Collection:
        """Get or create the shared memory collection."""
        try:
            collection = self._client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                embedding_function=self._embedding_fn,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                f"Collection '{self.COLLECTION_NAME}' ready. "
                f"Documents: {collection.count()}"
            )
            return collection
        except Exception as exc:
            logger.error(f"Failed to init collection: {exc}")
            raise

    # ------------------------------------------------------------------
    # Core Operations
    # ------------------------------------------------------------------

    def store(
        self,
        task_id: str,
        agent_name: str,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """
        Store a text document in the vector store.

        Parameters
        ----------
        task_id : str
            Workflow task identifier.
        agent_name : str
            Name of the agent producing this content.
        content : str
            Text to embed and store.
        metadata : dict, optional
            Additional metadata to attach.

        Returns
        -------
        str
            The generated document ID.
        """
        if not content or not content.strip():
            logger.warning(f"Skipping empty content from agent '{agent_name}'.")
            return ""

        doc_id = str(uuid.uuid4())
        doc_metadata = {
            "task_id": task_id,
            "agent_name": agent_name,
            "timestamp": str(time.time()),
            "token_estimate": str(estimate_tokens(content)),
        }
        if metadata:
            # Flatten: ChromaDB metadata values must be str/int/float/bool
            for k, v in metadata.items():
                doc_metadata[k] = str(v) if not isinstance(v, (str, int, float, bool)) else v

        with self._write_lock:
            try:
                self._collection.add(
                    documents=[content],
                    metadatas=[doc_metadata],
                    ids=[doc_id],
                )
                logger.info(
                    f"Stored doc {doc_id[:8]}... | "
                    f"agent={agent_name} | task={task_id} | "
                    f"tokens≈{estimate_tokens(content)}"
                )
                return doc_id
            except Exception as exc:
                logger.error(f"Failed to store document: {exc}")
                return ""

    def retrieve(
        self,
        task_id: str,
        query: str,
        top_k: int = 5,
        filter_agent: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Semantically retrieve the most relevant documents for a query.

        Parameters
        ----------
        task_id : str
            Filter results to this workflow.
        query : str
            Natural language query.
        top_k : int
            Number of results to return.
        filter_agent : str, optional
            If set, only return documents from this agent.

        Returns
        -------
        list[dict]
            List of {content, metadata, distance} dicts.
        """
        if not query.strip():
            return []

        where_filter: dict[str, Any] = {"task_id": {"$eq": task_id}}
        if filter_agent:
            where_filter["agent_name"] = {"$eq": filter_agent}

        try:
            total = self._collection.count()
            if total == 0:
                return []

            actual_k = min(top_k, total)
            results = self._collection.query(
                query_texts=[query],
                n_results=actual_k,
                where=where_filter,
            )

            entries = []
            if results and results.get("documents"):
                docs = results["documents"][0]
                metas = results.get("metadatas", [[]])[0]
                distances = results.get("distances", [[]])[0]
                for doc, meta, dist in zip(docs, metas, distances):
                    entries.append({
                        "content": doc,
                        "metadata": meta,
                        "distance": dist,
                    })

            logger.info(
                f"Retrieved {len(entries)} docs | "
                f"query='{query[:50]}' | task={task_id}"
            )
            return entries

        except Exception as exc:
            logger.error(f"Retrieval failed: {exc}")
            return []

    def get_conversation_history(
        self,
        task_id: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """
        Return all stored documents for a task, ordered by timestamp.

        Parameters
        ----------
        task_id : str
            Workflow task identifier.
        limit : int
            Maximum number of documents to return.
        """
        try:
            total = self._collection.count()
            if total == 0:
                return []

            results = self._collection.get(
                where={"task_id": {"$eq": task_id}},
            )

            entries = []
            if results and results.get("documents"):
                for doc, meta in zip(results["documents"], results.get("metadatas", [])):
                    entries.append({"content": doc, "metadata": meta})

            # Sort by timestamp ascending
            entries.sort(
                key=lambda x: float(x["metadata"].get("timestamp", "0"))
            )
            return entries[-limit:]

        except Exception as exc:
            logger.error(f"get_conversation_history failed: {exc}")
            return []

    def needs_summarization(self, task_id: str) -> bool:
        """Return True if stored history for task_id exceeds token threshold."""
        history = self.get_conversation_history(task_id)
        total = sum(estimate_tokens(e["content"]) for e in history)
        if total > MAX_TOKENS_CONTEXT:
            logger.info(
                f"Context window threshold reached: ~{total} tokens "
                f"for task {task_id}. Summarization needed."
            )
            return True
        return False

    def store_summary(self, task_id: str, summary: str) -> str:
        """Store a compressed summary document."""
        return self.store(
            task_id=task_id,
            agent_name="memory_agent",
            content=summary,
            metadata={"is_summary": True},
        )

    def clear_task(self, task_id: str) -> None:
        """Delete all documents associated with a specific task_id."""
        try:
            results = self._collection.get(
                where={"task_id": {"$eq": task_id}},
            )
            ids = results.get("ids", [])
            if ids:
                with self._write_lock:
                    self._collection.delete(ids=ids)
                logger.info(f"Cleared {len(ids)} docs for task {task_id}.")
        except Exception as exc:
            logger.error(f"Failed to clear task {task_id}: {exc}")

    def document_count(self) -> int:
        """Return total document count in the collection."""
        try:
            return self._collection.count()
        except Exception:
            return 0
