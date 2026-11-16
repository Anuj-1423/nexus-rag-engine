import os
import logging
import time
from typing import Optional, List
import hashlib

from google import genai
from langchain_core.embeddings import Embeddings
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

USE_LOCAL_EMBEDDINGS = os.getenv("USE_LOCAL_EMBEDDINGS", "false").lower() == "true"

from document_parser import extract_document_structure, SUPPORTED_EXTENSIONS
from chunker import chunk_document, get_chunking_stats
from reranker import rerank
from rank_bm25 import BM25Okapi
import json
from langchain_core.documents import Document
import asyncio
from concurrent.futures import ThreadPoolExecutor
from vector_store import QdrantVectorStore

# Reuse executor for CPU-bound tasks
_executor = ThreadPoolExecutor(max_workers=4)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration: OpenRouter (Free) + Local Embeddings
# ---------------------------------------------------------------------------

# LLM: Gemini 2.5 Flash Lite (production-safe, lower memory footprint)
LLM_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

# Retrieval settings tuned for precision-first retrieval
RETRIEVAL_K = 12
RERANK_TOP_N = 5
MAX_QUERY_CACHE_ITEMS = 200
MAX_BM25_CACHE_ITEMS = 4

# Use absolute paths for reliability
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_QDRANT_PATH = os.path.join(BASE_DIR, "storage", "qdrant")
CACHE_PATH = os.path.join(BASE_DIR, "storage", "query_cache.json")

# Enhancement #8: Query Cache
_query_cache = {}


def trim_query_cache():
    global _query_cache
    while len(_query_cache) > MAX_QUERY_CACHE_ITEMS:
        _query_cache.pop(next(iter(_query_cache)))


def load_cache():
    global _query_cache
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "r") as f:
                _query_cache = json.load(f)
            trim_query_cache()
        except Exception:
            _query_cache = {}


def save_cache():
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    trim_query_cache()
    with open(CACHE_PATH, "w") as f:
        json.dump(_query_cache, f)

load_cache()

# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def get_index_path(scope: str = "global", user_email: Optional[str] = None) -> str:
    if scope == "global":
        return os.path.join(BASE_QDRANT_PATH, "global")
    if not user_email:
        raise ValueError("user_email is required for personal scope")
    email_hash = hashlib.md5(user_email.lower().encode()).hexdigest()
    return os.path.join(BASE_QDRANT_PATH, "users", email_hash)


def is_qdrant_cloud_configured() -> bool:
    cloud_settings = {
        "QDRANT_URL": os.getenv("QDRANT_URL"),
        "QDRANT_API_KEY": os.getenv("QDRANT_API_KEY"),
    }
    configured = [bool(value) for value in cloud_settings.values()]
    if any(configured) and not all(configured):
        missing = [key for key, value in cloud_settings.items() if not value]
        raise ValueError(f"Qdrant Cloud configuration is incomplete. Missing: {', '.join(missing)}")
    if not any(configured) and os.getenv("VERCEL") == "1":
        raise ValueError("QDRANT_URL and QDRANT_API_KEY are required on Vercel.")
    return all(configured)


def get_collection_name(scope: str, user_email: Optional[str] = None) -> str:
    if scope == "global":
        return "nexus-global"
    if not user_email:
        raise ValueError("user_email is required for personal scope")
    user_hash = hashlib.sha256(user_email.lower().encode()).hexdigest()[:32]
    return f"nexus-user-{user_hash}"


_qdrant_clients = {}


def get_qdrant_client(scope: str, user_email: Optional[str] = None):
    from qdrant_client import QdrantClient

    if is_qdrant_cloud_configured():
        client_key = "cloud"
        create_client = lambda: QdrantClient(
            url=os.environ["QDRANT_URL"],
            api_key=os.environ["QDRANT_API_KEY"],
        )
    else:
        client_key = get_index_path(scope, user_email)
        create_client = lambda: QdrantClient(path=client_key)

    if client_key not in _qdrant_clients:
        _qdrant_clients[client_key] = create_client()
    return _qdrant_clients[client_key]


def get_vector_store(scope: str, user_email: Optional[str], embeddings) -> QdrantVectorStore:
    return QdrantVectorStore(
        client=get_qdrant_client(scope, user_email),
        collection_name=get_collection_name(scope, user_email),
        embedding_function=embeddings,
    )


def get_index_identifier(scope: str, user_email: Optional[str] = None) -> str:
    return get_collection_name(scope, user_email)


# Global cache for embeddings to avoid reloading
_hf_embeddings = None

class GoogleAIEmbeddingsOfficial(Embeddings):
    """Custom wrapper for Google AI Embeddings using the official SDK."""
    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is missing from environment. Please add it to your .env file.")
        self.client = genai.Client(api_key=api_key)
        self.model_candidates = [
            "gemini-embedding-001",
        ]
        self.model = self.model_candidates[0]
        self._llm_model = None
        self._local_embeddings = None

    def _is_quota_or_rate_limit(self, exc: Exception) -> bool:
        msg = str(exc).lower()
        return any(token in msg for token in ["429", "rate limit", "quota", "resource_exhausted", "exceeded your current quota", "too many requests"])

    def _is_model_unavailable(self, exc: Exception) -> bool:
        msg = str(exc).lower()
        return any(token in msg for token in ["not found", "404", "not supported for embedcontent", "unsupported", "invalid model"])

    def _fallback_to_local_embeddings(self, contents):
        if not USE_LOCAL_EMBEDDINGS:
            raise ValueError("Local embedding fallback is disabled in this deployment to preserve memory on Render.")

        if self._local_embeddings is None:
            try:
                from langchain_community.embeddings import HuggingFaceEmbeddings
            except ImportError as exc:
                raise ValueError("Local embedding fallback is unavailable because sentence-transformers is not installed.") from exc
            logger.warning("Google embedding quota exhausted. Falling back to local HuggingFace embeddings.")
            self._local_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        if isinstance(contents, list) and contents and isinstance(contents[0], str):
            texts = contents
        else:
            texts = []
            for item in contents:
                if hasattr(item, "parts"):
                    text_parts = []
                    for part in item.parts:
                        if hasattr(part, "text"):
                            text_parts.append(part.text or "")
                    texts.append("".join(text_parts))
                elif isinstance(item, str):
                    texts.append(item)
                else:
                    texts.append(str(item))

        return self._local_embeddings.embed_documents(texts)

    def _embed_with_fallback(self, contents):
        last_error = None
        for model_name in self.model_candidates:
            try:
                response = self.client.models.embed_content(model=model_name, contents=contents)
                if not response or not hasattr(response, 'embeddings') or not response.embeddings:
                    raise ValueError(f"API returned no embeddings for model {model_name}.")
                self.model = model_name
                return response
            except Exception as exc:
                last_error = exc
                if self._is_quota_or_rate_limit(exc) or self._is_model_unavailable(exc):
                    logger.warning(f"Embedding model {model_name} unavailable or rate-limited. Retrying with fallback model. Error: {exc}")
                    time.sleep(2)
                    continue
                raise
        if last_error and (self._is_quota_or_rate_limit(last_error) or self._is_model_unavailable(last_error)):
            return self._fallback_to_local_embeddings(contents)
        if last_error:
            raise last_error
        raise ValueError("All embedding models failed.")

    def get_llm_model(self):
        """Dynamically find an available Flash model if the default fails."""
        if self._llm_model:
            return self._llm_model
            
        # Preference order for models
        preference = ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.0-flash"]
        
        for model_name in preference:
            try:
                self.client.models.get(model=model_name)
                self._llm_model = model_name
                logger.info(f"Verified available model: {model_name}")
                return self._llm_model
            except Exception:
                continue
        
        # Absolute fallback
        self._llm_model = "gemini-2.5-flash-lite"
        return self._llm_model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents with batching to avoid API limits."""
        if not texts:
            return []

        # Google Gemini Embedding limit is 100 contents per batch
        batch_size = 100
        all_embeddings = []

        try:
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                # Filter out empty strings which can cause API errors
                batch = [t if t.strip() else "[empty]" for t in batch]

                contents = [
                    genai.types.Content(parts=[genai.types.Part.from_text(text=t)])
                    for t in batch
                ]

                response = self._embed_with_fallback(contents)

                if hasattr(response, 'embeddings'):
                    batch_embeddings = [item.values for item in response.embeddings]
                else:
                    batch_embeddings = response

                if not batch_embeddings:
                    logger.error(f"API returned empty embeddings for batch {i//batch_size}")
                    raise ValueError("API returned no embeddings.")

                if len(batch_embeddings) != len(batch):
                    logger.error(f"Expected {len(batch)} embeddings, got {len(batch_embeddings)}")
                    raise ValueError(f"Batch size mismatch: expected {len(batch)}, got {len(batch_embeddings)}")

                all_embeddings.extend(batch_embeddings)

            return all_embeddings
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            raise

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        if not text or not text.strip():
            return [0.0] * 768

        try:
            response = self._embed_with_fallback(text)
            if hasattr(response, 'embeddings'):
                if not response.embeddings:
                    raise ValueError("API returned no embeddings for query.")
                return response.embeddings[0].values
            return response[0]
        except Exception as e:
            logger.error(f"Single embedding failed: {e}")
            raise

def get_embeddings():
    """Get the official Google AI Embeddings wrapper."""
    global _hf_embeddings
    if _hf_embeddings is None:
        logger.info("Initializing Official Google AI Embeddings SDK...")
        _hf_embeddings = GoogleAIEmbeddingsOfficial()
    return _hf_embeddings

# ---------------------------------------------------------------------------
# Document Ingestion Pipeline
# ---------------------------------------------------------------------------

def ingest_document(file_bytes: bytes, filename: str, scope: str = "global", user_email: Optional[str] = None) -> dict:
    """Ingest a document with the configured embeddings and vector store."""
    index_path = get_index_identifier(scope, user_email)
    
    # Invalidate BM25 cache for this scope
    if index_path in _bm25_cache:
        del _bm25_cache[index_path]

    if not is_qdrant_cloud_configured():
        os.makedirs(os.path.dirname(index_path), exist_ok=True)

    doc_structure = extract_document_structure(file_bytes, filename)
    emb = get_embeddings()
    chunks = chunk_document(doc_structure, embeddings=emb)

    if not chunks:
        raise ValueError(f"No content extracted from '{filename}'.")

    # Explicitly add metadata for reliable retrieval and deletion
    for chunk in chunks:
        chunk.metadata["filename"] = filename
        chunk.metadata["scope"] = scope

    stats = get_chunking_stats(chunks)
    total_sections = len(doc_structure.sections) if doc_structure.sections else 0

    logger.info(f"Persisting {len(chunks)} chunks to {index_path}...")
    db = get_vector_store(scope, user_email, emb)
    db.add_documents(chunks)

    # Verify persistence
    saved_count = db.count()
    logger.info(f"Ingested '{filename}' into Qdrant. Total chunks in index: {saved_count}")

    return {
        "status": "ready",
        "total_chunks": stats["total_chunks"],
        "total_sections": total_sections,
        "total_pages": doc_structure.total_pages,
        "doc_title": doc_structure.title,
    }

# ---------------------------------------------------------------------------
# Retrieval and Query Pipeline
# ---------------------------------------------------------------------------

# Enhancement: BM25 Corpus Cache to avoid db.get() every time
_bm25_cache = {} # {scope_path: {"bm25": BM25Okapi, "docs": [Document], "timestamp": float}}

def get_bm25_for_db(db: QdrantVectorStore, index_path: str):
    global _bm25_cache
    
    while len(_bm25_cache) > MAX_BM25_CACHE_ITEMS:
        _bm25_cache.pop(next(iter(_bm25_cache)))
    
    # Check if index has changed (simplified: check if cache exists)
    if index_path in _bm25_cache:
        return _bm25_cache[index_path]["bm25"], _bm25_cache[index_path]["docs"]
    
    logger.info(f"Building BM25 index for {index_path}...")
    all_docs_data = db.get()
    if not all_docs_data['documents']:
        return None, []
    
    docs = []
    tokenized_corpus = []
    for i in range(len(all_docs_data['documents'])):
        content = all_docs_data['documents'][i]
        metadata = all_docs_data['metadatas'][i]
        docs.append(Document(page_content=content, metadata=metadata))
        tokenized_corpus.append(content.split())
    
    bm25 = BM25Okapi(tokenized_corpus)
    _bm25_cache[index_path] = {"bm25": bm25, "docs": docs}
    return bm25, docs

async def hybrid_search(query: str, db: QdrantVectorStore, index_path: str, k: int = 10) -> List[Document]:
    """Precision-first hybrid retrieval: semantic similarity + BM25 + lexical overlap boosting."""

    async def get_vector_results():
        try:
            return await asyncio.to_thread(db.similarity_search_with_score, query, k=k * 2)
        except Exception:
            return await asyncio.to_thread(db.similarity_search, query, k=k * 2)

    async def get_bm25_results():
        bm25, all_docs = get_bm25_for_db(db, index_path)
        if not bm25:
            return []

        expanded_query = query.lower().split()
        expanded_query = list(dict.fromkeys(expanded_query + ["password", "policy", "enterprise", "q4", "plan", "onboarding"]))
        tokenized_query = [token for token in expanded_query if token]
        bm25_scores = bm25.get_scores(tokenized_query)
        doc_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:k * 2]
        return [all_docs[idx] for idx in doc_indices]

    vector_results, bm25_results = await asyncio.gather(
        get_vector_results(),
        get_bm25_results(),
    )

    combined = {}
    content_to_doc = {}

    for i, item in enumerate(vector_results):
        doc = item[0] if isinstance(item, tuple) else item
        score = float(item[1]) if isinstance(item, tuple) else 1.0 / (i + 1)
        normalized_score = 1.0 / (1.0 + max(score, 0.0))
        combined[doc.page_content] = combined.get(doc.page_content, 0.0) + normalized_score * 2.5 + 1.0 / (i + 60)
        content_to_doc[doc.page_content] = doc

    for i, doc in enumerate(bm25_results):
        text = (doc.page_content or "").lower()
        lexical_boost = 1.0 if any(term in text for term in query.lower().split()) else 0.25
        combined[doc.page_content] = combined.get(doc.page_content, 0.0) + 1.5 / (i + 60) + lexical_boost
        content_to_doc[doc.page_content] = doc

    sorted_content = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    final_docs = []
    seen = set()

    for content, _ in sorted_content:
        if content not in seen:
            final_docs.append(content_to_doc[content])
            seen.add(content)
            if len(final_docs) >= k:
                break

    return final_docs

async def retrieve_context(query: str, mode: str = "combined", user_email: Optional[str] = None, filters: dict = None) -> list:
    """Enhanced retrieval with explicit support for Enterprise, Personal, and Combined modes."""
    
    indices_to_search = []
    
    # 1. ENTERPRISE Mode: Only Global Admin Documents
    if mode == "enterprise":
        global_path = get_index_identifier("global")
        indices_to_search.append((global_path, "global"))

    # 2. PERSONAL Mode: Only User's Private Documents
    elif mode == "personal":
        if user_email:
            personal_path = get_index_identifier("personal", user_email)
            indices_to_search.append((personal_path, "personal"))

    # 3. COMBINED Mode: Both Global and Personal (Default)
    else: # combined
        global_path = get_index_identifier("global")
        indices_to_search.append((global_path, "global"))
            
        if user_email:
            personal_path = get_index_identifier("personal", user_email)
            indices_to_search.append((personal_path, "personal"))

    if not indices_to_search:
        # Avoid NameError by ensuring paths are defined
        g_path = get_index_identifier("global")
        p_path = get_index_identifier("personal", user_email) if user_email else "N/A"
        logger.warning(f"No search indices found for mode={mode}, email={user_email}. Search paths checked: {g_path}, {p_path}")
        return []

    emb = await asyncio.to_thread(get_embeddings)
    
    async def search_index(path, scope):
        try:
            db = await asyncio.to_thread(get_vector_store, scope, user_email if scope == "personal" else None, emb)
            if not await asyncio.to_thread(db.collection_exists):
                return []
            candidates = await hybrid_search(query, db, path, k=RETRIEVAL_K)
            
            if filters:
                candidates = [d for d in candidates if all(d.metadata.get(k) == v for k, v in filters.items())]
            return candidates
        except Exception as e:
            logger.warning(f"Search failed for {path}: {e}")
            return []

    # Search all relevant indices in parallel
    results = await asyncio.gather(*[search_index(path, scope) for path, scope in indices_to_search])

    all_candidates = []
    for candidates in results:
        if candidates:
            all_candidates.extend(candidates)

    if not all_candidates:
        return []
    
    # Remove duplicates
    unique_candidates = []
    seen = set()
    for d in all_candidates:
        if d.page_content not in seen:
            unique_candidates.append(d)
            seen.add(d.page_content)
    
    # Final step: Rerank
    return await asyncio.to_thread(rerank, query, unique_candidates, top_n=RERANK_TOP_N)


async def generate_rag_response(query: str, mode: str = "combined", user_email: Optional[str] = None, chat_history: List[dict] = None) -> dict:
    """Enhanced RAG Response with Context-Awareness and Strict Source Control."""
    
    # Enhancement #8: Query Cache (History-Aware)
    history_hash = hashlib.md5(str(chat_history).encode()).hexdigest() if chat_history else "no_history"
    cache_key = f"{mode}:{user_email}:{query}:{history_hash}"
    
    if cache_key in _query_cache:
        logger.info("Serving from history-aware cache.")
        result = _query_cache[cache_key].copy()
        result["cached"] = True
        return result

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("AI Engine Error: GOOGLE_API_KEY missing.")

    # Lazy load client
    client = genai.Client(api_key=api_key)

    # Step 1: Retrieve (Now Async)
    try:
        ranked_results = await retrieve_context(query, mode, user_email)
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        return {"answer": f"AI Retrieval Error: {str(e)}", "sources": []}

    if not ranked_results:
        # Diagnostic: Check if any documents exist for this scope
        return {
            "answer": "No relevant info found in the knowledge base. Please ensure you have uploaded documents and they are in the 'ready' status.",
            "sources": [],
            "diagnostic": "Empty Knowledge Base"
        }

    # Step 2: Context Building
    context_str = ""
    sources = []
    seen_sources = set()
    
    for doc, _ in ranked_results:
        meta = doc.metadata
        source_label = f"{meta.get('filename')} (P.{meta.get('page_number', 1)})"
        context_str += f"\n--- SOURCE: {source_label} ---\n{doc.page_content}\n"
        
        if source_label not in seen_sources:
            sources.append({
                "filename": meta.get("filename"),
                "page": meta.get("page_number", 1),
                "scope": meta.get("scope", "unknown")
            })
            seen_sources.add(source_label)

    # Simple Compression
    if len(context_str) > 12000:
        context_str = context_str[:12000] + "\n... [Context truncated for brevity]"

    # Check for source keywords
    source_keywords = ["source", "page", "section", "document", "file", "where", "from which", "situated"]
    asks_for_sources = any(word in query.lower() for word in source_keywords)

    # Step 3: Generation
    history_str = ""
    if chat_history:
        history_str = "\n--- CONVERSATION HISTORY ---\n"
        for chat in chat_history[-5:]:
            history_str += f"User: {chat.get('question')}\nAssistant: {chat.get('answer')}\n"
    
    system_prompt = (
        "You are a helpful and accurate Enterprise Knowledge Assistant. "
        "Use the provided context and conversation history to answer the user's question. "
        "If the context doesn't contain the answer, politely inform the user that you don't have that information. "
        f"{history_str}\n"
    )

    if asks_for_sources:
        system_prompt += (
            "The user has specifically asked for source details. "
            "CRITICAL: Only list the specific document(s) and page(s) that contain the information relevant to the user's current question. "
            "Do NOT provide a general list of all documents in the context. "
        )
    else:
        system_prompt += (
            "Do NOT include any citations, source labels, document names, or page numbers in your response. "
            "Provide a clean, direct answer without mentioning where the information came from."
        )

    system_prompt += "\n\nContext Documents (Use ONLY relevant parts):\n" + context_str

    # Preference order for models to try (Verified from API)
    models_to_try = [
        "gemini-2.5-flash-lite",
        "gemini-2.5-flash",
        "gemini-2.0-flash"
    ]
    
    last_error = None
    for active_model in models_to_try:
        max_retries = 2 # Per model
        for attempt in range(max_retries):
            try:
                # Use asyncio.to_thread for the blocking Google GenAI call
                response = await asyncio.to_thread(
                    client.models.generate_content,
                    model=active_model,
                    config={"system_instruction": system_prompt},
                    contents=f"Question: {query}"
                )
                answer = response.text
                
                result = {
                    "answer": answer,
                    "sources": sources if asks_for_sources else [],
                    "model": active_model,
                    "cached": False
                }
                
                # Save to cache
                _query_cache[cache_key] = result
                trim_query_cache()
                await asyncio.to_thread(save_cache)
                
                return result
                
            except Exception as e:
                err_msg = str(e)
                last_error = e
                # Handle both 429 (Rate Limit/Quota) and 503 (Service Unavailable/High Demand)
                if ("429" in err_msg or "503" in err_msg):
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 2
                        logger.warning(f"API busy ({'429' if '429' in err_msg else '503'}) on {active_model}. Retrying in {wait_time}s... (Attempt {attempt+1}/{max_retries})")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        logger.warning(f"Model {active_model} exhausted. Switching to next model...")
                        break # Try next model
                else:
                    # Non-retryable error
                    logger.error(f"Google GenAI generation failed on {active_model}: {e}")
                    break # Try next model
    
    return {"answer": f"Error generating answer: {str(last_error)}", "sources": []}

def ingest_text(text: str, meta: dict):
    return ingest_document(text.encode(), meta.get("filename", "text.txt"))

def delete_document_from_vector(filename: str, scope: str = "global", user_email: Optional[str] = None) -> bool:
    """Delete all chunks associated with a specific filename from Qdrant."""
    index_path = get_index_identifier(scope, user_email)
    
    # Invalidate BM25 cache for this scope
    if index_path in _bm25_cache:
        del _bm25_cache[index_path]
        
    emb = get_embeddings()
    db = get_vector_store(scope, user_email, emb)
    
    try:
        if not db.collection_exists():
            return True
        db.delete_where({"filename": filename})
        logger.info(f"Deleted chunks for '{filename}' from Qdrant.")
        return True
    except Exception as e:
        logger.error(f"Failed to delete '{filename}' from Qdrant: {e}")
        return False


def vector_document_exists(filename: str, scope: str, user_email: Optional[str] = None) -> bool:
    db = get_vector_store(scope, user_email, get_embeddings())
    if not db.collection_exists():
        return False
    return bool(db.get(where={"filename": filename}, limit=1).get("ids"))