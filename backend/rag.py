import os
import logging
from typing import Optional, List
import hashlib

from google import genai
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma

from document_parser import extract_document_structure, SUPPORTED_EXTENSIONS
from chunker import chunk_document, get_chunking_stats
from reranker import rerank
from rank_bm25 import BM25Okapi
import json
from langchain_core.documents import Document
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Reuse executor for CPU-bound tasks
_executor = ThreadPoolExecutor(max_workers=4)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration: OpenRouter (Free) + Local Embeddings
# ---------------------------------------------------------------------------

# LLM: Gemini 2.5 Flash (Current stable)
LLM_MODEL = "gemini-2.5-flash"

# Retrieval settings
RETRIEVAL_K = 20
RERANK_TOP_N = 8

BASE_CHROMA_PATH = "storage/chroma"
CACHE_PATH = "storage/query_cache.json"

# Enhancement #8: Query Cache
_query_cache = {}

def load_cache():
    global _query_cache
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "r") as f:
                _query_cache = json.load(f)
        except:
            _query_cache = {}

def save_cache():
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(_query_cache, f)

load_cache()

# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def get_index_path(scope: str = "global", user_email: Optional[str] = None) -> str:
    if scope == "global":
        return os.path.join(BASE_CHROMA_PATH, "global")
    if not user_email:
        raise ValueError("user_email is required for personal scope")
    email_hash = hashlib.md5(user_email.lower().encode()).hexdigest()
    return os.path.join(BASE_CHROMA_PATH, "users", email_hash)

# Global cache for embeddings to avoid reloading
_hf_embeddings = None

class GoogleAIEmbeddingsOfficial(Embeddings):
    """Custom wrapper for Google AI Embeddings using the official SDK."""
    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        self.client = genai.Client(api_key=api_key)
        self.model = "models/gemini-embedding-2"
        self._llm_model = None

    def get_llm_model(self):
        """Dynamically find an available Flash model if the default fails."""
        if self._llm_model:
            return self._llm_model
            
        default_model = "gemini-1.5-flash"
        try:
            # Try to verify the default model exists
            self.client.models.get(model=default_model)
            self._llm_model = default_model
            return self._llm_model
        except Exception:
            try:
                # List available models and find a Flash model
                models = self.client.models.list()
                for m in models:
                    # The correct attribute in google-genai is 'supported_actions'
                    if m.supported_actions and "generateContent" in m.supported_actions and "flash" in m.name.lower():
                        logger.info(f"Dynamically selected model: {m.name}")
                        # Store name without 'models/' prefix if present
                        self._llm_model = m.name.replace("models/", "")
                        return self._llm_model
            except Exception as e:
                logger.error(f"Failed to list models: {e}")
        
        return default_model # Fallback

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
                
                # Wrap each string in a Content object to get individual embeddings
                # Otherwise, the SDK wraps the list into a single Content with multiple Parts
                contents = [
                    genai.types.Content(parts=[genai.types.Part.from_text(text=t)])
                    for t in batch
                ]

                response = self.client.models.embed_content(
                    model=self.model,
                    contents=contents
                )
                
                if not response or not hasattr(response, 'embeddings') or not response.embeddings:
                    logger.error(f"API returned empty embeddings for batch {i//batch_size}")
                    raise ValueError("API returned no embeddings.")

                batch_embeddings = [item.values for item in response.embeddings]
                
                # Check if we got the expected number of embeddings
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
            # Single string is automatically wrapped by the SDK
            response = self.client.models.embed_content(
                model=self.model,
                contents=text
            )
            if not response or not response.embeddings:
                 raise ValueError("API returned no embeddings for query.")
                 
            return response.embeddings[0].values
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
    """Ingest document using Local Embeddings."""
    index_path = get_index_path(scope, user_email)
    
    # Invalidate BM25 cache for this scope
    if index_path in _bm25_cache:
        del _bm25_cache[index_path]
        
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

    # Chroma handles persistence automatically. If the directory exists, it loads and appends.
    db = Chroma.from_documents(chunks, emb, persist_directory=index_path)

    logger.info(f"Ingested '{filename}' into ChromaDB using Local HF Embeddings.")

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

def get_bm25_for_db(db: Chroma, index_path: str):
    global _bm25_cache
    
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

async def hybrid_search(query: str, db: Chroma, index_path: str, k: int = 10) -> List[Document]:
    """Combine Chroma vector search with BM25 keyword search in parallel."""
    
    # 1. Run Vector Search and BM25 Fetching in parallel
    async def get_vector_results():
        return await asyncio.to_thread(db.similarity_search, query, k=k*2)

    async def get_bm25_results():
        bm25, all_docs = get_bm25_for_db(db, index_path)
        if not bm25:
            return []
        
        tokenized_query = query.split()
        bm25_scores = bm25.get_scores(tokenized_query)
        doc_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:k*2]
        
        results = []
        for idx in doc_indices:
            results.append(all_docs[idx])
        return results

    # Execute in parallel
    vector_results, bm25_results = await asyncio.gather(
        get_vector_results(),
        get_bm25_results()
    )
    
    if not bm25_results:
        return vector_results[:k]
    
    # 2. Combine (Reciprocal Rank Fusion - Simplified)
    combined = {}
    content_to_doc = {}
    
    for i, doc in enumerate(vector_results):
        combined[doc.page_content] = combined.get(doc.page_content, 0) + 1 / (i + 60)
        content_to_doc[doc.page_content] = doc
    
    for i, doc in enumerate(bm25_results):
        combined[doc.page_content] = combined.get(doc.page_content, 0) + 1 / (i + 60)
        if doc.page_content not in content_to_doc:
            content_to_doc[doc.page_content] = doc
        
    # Sort and return unique docs
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
        global_path = get_index_path("global")
        if os.path.exists(global_path):
            indices_to_search.append((global_path, "global"))

    # 2. PERSONAL Mode: Only User's Private Documents
    elif mode == "personal":
        if user_email:
            personal_path = get_index_path("personal", user_email)
            if os.path.exists(personal_path):
                indices_to_search.append((personal_path, "personal"))

    # 3. COMBINED Mode: Both Global and Personal (Default)
    else: # combined
        global_path = get_index_path("global")
        if os.path.exists(global_path):
            indices_to_search.append((global_path, "global"))
            
        if user_email:
            personal_path = get_index_path("personal", user_email)
            if os.path.exists(personal_path):
                indices_to_search.append((personal_path, "personal"))

    if not indices_to_search:
        return []

    emb = await asyncio.to_thread(get_embeddings)
    
    async def search_index(path, scope):
        try:
            db = await asyncio.to_thread(Chroma, persist_directory=path, embedding_function=emb)
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
    ranked_results = await retrieve_context(query, mode, user_email)
    if not ranked_results:
        return {"answer": "No relevant info found in the knowledge base.", "sources": []}

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

    # LLM_MODEL changed to "gemini-1.5-flash-latest" at top of file
    
    max_retries = 3
    # Use dynamic model discovery
    active_model = get_embeddings().get_llm_model()
    
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
            await asyncio.to_thread(save_cache)
            
            return result
            
        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg and attempt < max_retries - 1:
                wait_time = (attempt + 1) * 5 # Exponential-ish backoff
                logger.warning(f"Rate limited (429) on {active_model}. Retrying in {wait_time}s... (Attempt {attempt+1}/{max_retries})")
                await asyncio.sleep(wait_time)
                continue
                
            logger.error(f"Google GenAI generation failed (Attempt {attempt+1}): {e}")
            return {"answer": f"Error generating answer: {e}", "sources": []}

def ingest_text(text: str, meta: dict):
    return ingest_document(text.encode(), meta.get("filename", "text.txt"))

def delete_document_from_vector(filename: str, scope: str = "global", user_email: Optional[str] = None) -> bool:
    """Delete all chunks associated with a specific filename from ChromaDB."""
    index_path = get_index_path(scope, user_email)
    
    # Invalidate BM25 cache for this scope
    if index_path in _bm25_cache:
        del _bm25_cache[index_path]
        
    if not os.path.exists(index_path):
        logger.warning(f"Cannot delete {filename}: Index path {index_path} does not exist.")
        return False
        
    emb = get_embeddings()
    db = Chroma(persist_directory=index_path, embedding_function=emb)
    
    try:
        # We need to get the ids of documents with this filename
        result = db.get(where={"filename": filename})
        ids_to_delete = result.get('ids', [])
        
        if not ids_to_delete:
            logger.info(f"No chunks found for '{filename}' in {scope} database.")
            return True # Nothing to delete
            
        db.delete(ids_to_delete)
        logger.info(f"Deleted {len(ids_to_delete)} chunks for '{filename}' from ChromaDB.")
        return True
    except Exception as e:
        logger.error(f"Failed to delete '{filename}' from ChromaDB: {e}")
        return False