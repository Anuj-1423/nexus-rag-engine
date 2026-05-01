"""
Re-Ranking Module (API-Based)
=============================
Uses Cohere's Rerank API to provide high-precision document ranking
without consuming server RAM. 100% compatible with Render Free Tier.
"""

import os
import logging
import requests
from typing import Optional, List
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Configuration
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
RERANK_MODEL = "rerank-english-v3.0"  # High performance model

def rerank(
    query: str,
    documents: List[Document],
    top_n: int = 4,
) -> List[tuple[Document, float]]:
    """
    Re-rank documents using Cohere's API.
    
    Args:
        query: User search query
        documents: List of candidate LangChain Documents
        top_n: Number of results to return
    """
    if not documents:
        return []

    if not COHERE_API_KEY:
        logger.warning("COHERE_API_KEY not found. Skipping re-ranking and returning original results.")
        return [(doc, 0.0) for doc in documents[:top_n]]

    try:
        # Prepare docs for Cohere API (extracting text)
        doc_texts = [doc.page_content for doc in documents]
        
        url = "https://api.cohere.ai/v1/rerank"
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {COHERE_API_KEY}"
        }
        payload = {
            "model": RERANK_MODEL,
            "query": query,
            "documents": doc_texts,
            "top_n": top_n
        }

        response = requests.post(url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        
        results = response.json().get("results", [])
        
        # Map API results back to LangChain Documents
        ranked_docs = []
        for res in results:
            idx = res["index"]
            score = res["relevance_score"]
            ranked_docs.append((documents[idx], score))
            
        return ranked_docs

    except Exception as e:
        logger.error(f"Cohere Re-ranking failed: {e}")
        # Fallback to original order
        return [(doc, 0.0) for doc in documents[:top_n]]

def rerank_simple(
    query: str,
    documents: List[Document],
    top_n: int = 4,
) -> List[Document]:
    """Convenience wrapper that returns only Document objects."""
    ranked = rerank(query, documents, top_n=top_n)
    return [doc for doc, _score in ranked]
