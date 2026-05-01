"""
Re-Ranking Module
=================
Cross-encoder re-ranking for improved retrieval precision.
Uses sentence-transformers with a lightweight cross-encoder model
to score (query, document) pairs and return the most relevant results.
"""

import logging
from typing import Optional

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Lazy-loaded singleton — avoids startup cost
_reranker_model = None


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------

def _load_reranker():
    """Lazy-load the cross-encoder model on first use."""
    global _reranker_model
    if _reranker_model is not None:
        return _reranker_model

    try:
        from sentence_transformers import CrossEncoder
        logger.info(f"Loading re-ranker model: {RERANKER_MODEL_NAME}")
        _reranker_model = CrossEncoder(RERANKER_MODEL_NAME)
        logger.info("Re-ranker model loaded successfully.")
        return _reranker_model
    except Exception as e:
        logger.warning(
            f"Failed to load re-ranker model: {e}. "
            "Falling back to original retrieval order."
        )
        return None


# ---------------------------------------------------------------------------
# Re-Ranking
# ---------------------------------------------------------------------------

def rerank(
    query: str,
    documents: list[Document],
    top_n: int = 4,
    min_score: Optional[float] = None,
) -> list[tuple[Document, float]]:
    """
    Re-rank retrieved documents using the cross-encoder model.
    
    Each (query, document.page_content) pair is scored by the cross-encoder,
    and the documents are returned sorted by relevance score (descending).
    
    Args:
        query: The user's search query
        documents: List of candidate documents from initial retrieval
        top_n: Number of top documents to return
        min_score: Optional minimum score threshold; documents below this
                   are filtered out
    
    Returns:
        List of (Document, score) tuples, sorted by descending relevance.
        Falls back to original order with score=0.0 if model is unavailable.
    """
    if not documents:
        return []

    model = _load_reranker()

    if model is None:
        # Fallback: return original order with neutral scores
        return [(doc, 0.0) for doc in documents[:top_n]]

    # Build query-document pairs for scoring
    pairs = [(query, doc.page_content) for doc in documents]

    try:
        scores = model.predict(pairs)
    except Exception as e:
        logger.error(f"Re-ranking prediction failed: {e}")
        return [(doc, 0.0) for doc in documents[:top_n]]

    # Pair documents with their scores and sort descending
    scored_docs = list(zip(documents, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    # Apply minimum score filter if specified
    if min_score is not None:
        scored_docs = [(doc, s) for doc, s in scored_docs if s >= min_score]

    # Return top-N
    return scored_docs[:top_n]


def rerank_simple(
    query: str,
    documents: list[Document],
    top_n: int = 4,
) -> list[Document]:
    """
    Convenience wrapper: returns just the re-ranked documents without scores.
    """
    ranked = rerank(query, documents, top_n=top_n)
    return [doc for doc, _score in ranked]
