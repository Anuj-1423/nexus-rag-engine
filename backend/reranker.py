"""
Re-Ranking Module (API-Based)
=============================
Uses Cohere's Rerank API to provide high-precision document ranking
without consuming server RAM. 100% compatible with Render Free Tier.
"""

import json
import logging
import os
import re
from typing import List, Set

from dotenv import load_dotenv
import requests
from langchain_core.documents import Document

load_dotenv()

logger = logging.getLogger(__name__)

# Configuration
COHERE_API_KEY = os.getenv("COHERE_API_KEY") or os.getenv("cohere_api_key")
RERANK_MODEL = "rerank-english-v3.0"

_QUERY_SYNONYMS = {
    "password": {"password", "credentials", "auth", "login", "secret"},
    "policy": {"policy", "rules", "procedure", "standard", "requirement"},
    "enterprise": {"enterprise", "company", "organization", "business"},
    "q4": {"q4", "quarter4", "fourth quarter"},
    "upload": {"upload", "uploaded", "submitted", "added", "stored"},
    "plan": {"plan", "package", "offer", "program"},
    "onboarding": {"onboarding", "welcome", "setup", "orientation"},
    "include": {"include", "includes", "contains", "covers", "encompasses"},
}


def _normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def _tokenize(text: str) -> List[str]:
    return [token for token in _normalize_text(text).split() if token]


def _expand_query_terms(query: str) -> Set[str]:
    tokens = set(_tokenize(query))
    expanded = set(tokens)
    for token in list(tokens):
        for alias, variants in _QUERY_SYNONYMS.items():
            if token == alias or token in variants:
                expanded |= {alias, *variants}
                expanded.add(token)
    return expanded


def _doc_text(doc: Document) -> str:
    return " ".join([
        doc.page_content or "",
        doc.metadata.get("filename", ""),
        doc.metadata.get("section_heading", ""),
        doc.metadata.get("doc_title", ""),
    ])


def _lexical_score(query: str, doc: Document) -> float:
    query_terms = _expand_query_terms(query)
    if not query_terms:
        return 0.0

    doc_text = _doc_text(doc)
    doc_tokens = _tokenize(doc_text)
    doc_set = set(doc_tokens)
    query_text = query.lower().strip()

    overlap = len(query_terms & doc_set)
    tf_bonus = sum(doc_tokens.count(term) for term in query_terms)
    exact_phrase = 1.0 if query_text and query_text in doc_text.lower() else 0.0
    phrase_overlap = sum(1 for term in query_terms if term in doc_text.lower())

    filename = (doc.metadata.get("filename") or "").lower()
    section = (doc.metadata.get("section_heading") or "").lower()
    title = (doc.metadata.get("doc_title") or "").lower()

    filename_bonus = 3.0 if any(term in filename for term in query_terms) else 0.0
    section_bonus = 2.0 if any(term in section for term in query_terms) else 0.0
    title_bonus = 2.5 if any(term in title for term in query_terms) else 0.0
    adjacency_bonus = 2.0 if query_text and query_text.replace(" ", "") in doc_text.lower().replace(" ", "") else 0.0

    return (
        overlap * 6.0
        + tf_bonus * 1.8
        + phrase_overlap * 1.4
        + exact_phrase * 5.0
        + filename_bonus
        + section_bonus
        + title_bonus
        + adjacency_bonus
    )


def _fallback_rerank(query: str, documents: List[Document], top_n: int) -> List[tuple[Document, float]]:
    scored = [(doc, _lexical_score(query, doc)) for doc in documents]
    scored.sort(key=lambda item: item[1], reverse=True)
    return [(doc, score) for doc, score in scored[:top_n]]


def rerank(
    query: str,
    documents: List[Document],
    top_n: int = 4,
) -> List[tuple[Document, float]]:
    """Precision-first reranking: use Cohere when available, else a stronger lexical fallback."""
    if not documents:
        return []

    if not COHERE_API_KEY:
        logger.warning("COHERE_API_KEY not found. Using local lexical re-ranking fallback.")
        return _fallback_rerank(query, documents, top_n)

    try:
        doc_texts = [doc.page_content for doc in documents]
        url = "https://api.cohere.ai/v1/rerank"
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {COHERE_API_KEY}",
        }
        payload = {
            "model": RERANK_MODEL,
            "query": query,
            "documents": doc_texts,
            "top_n": top_n,
        }

        response = requests.post(url, json=payload, headers=headers, timeout=15)
        response.raise_for_status()

        results = response.json().get("results", [])
        ranked_docs = []
        for res in results:
            idx = res["index"]
            score = res["relevance_score"]
            ranked_docs.append((documents[idx], score))

        if ranked_docs:
            return ranked_docs
        raise ValueError("No results from Cohere rerank API")

    except Exception as e:
        logger.error(f"Cohere Re-ranking failed: {e}")
        logger.warning("Falling back to local lexical re-ranking.")
        return _fallback_rerank(query, documents, top_n)


def rerank_simple(
    query: str,
    documents: List[Document],
    top_n: int = 4,
) -> List[Document]:
    """Convenience wrapper that returns only Document objects."""
    ranked = rerank(query, documents, top_n=top_n)
    return [doc for doc, _score in ranked]
def rerank_simple(
    query: str,
    documents: List[Document],
    top_n: int = 4,
) -> List[Document]:
    """Convenience wrapper that returns only Document objects."""
    ranked = rerank(query, documents, top_n=top_n)
    return [doc for doc, _score in ranked]
