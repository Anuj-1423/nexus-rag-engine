import os
import logging
from typing import Optional, List
import hashlib

from google import genai
from langchain_huggingface import HuggingFaceEmbeddings
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

# LLM: Gemini Flash (via Google API)
LLM_MODEL = "gemini-flash-latest"

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

def get_embeddings():
    """Get Local HuggingFace Embeddings (100% Free, No Key Needed)."""
    global _hf_embeddings
    if _hf_embeddings is None:
        logger.info("Loading Local HuggingFace Embeddings (this may take a moment on first run)...")
        # Uses 'all-MiniLM-L6-v2' which is fast and accurate for RAG
        _hf_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
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

    # Explicitly add filename to metadata for reliable deletion later
    for chunk in chunks:
        chunk.metadata["filename"] = filename

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

async def retrieve_context(query: str, scope: str = "global", user_email: Optional[str] = None, filters: dict = None) -> list:
    """Enhanced retrieval with Hybrid Search and Metadata Filtering."""
    index_path = get_index_path(scope, user_email)
    if not os.path.exists(index_path):
        return []

    emb = await asyncio.to_thread(get_embeddings)
    db = Chroma(persist_directory=index_path, embedding_function=emb)
    
    # Enhancement #5: Metadata Filtering
    try:
        # Use hybrid search (now async)
        candidates = await hybrid_search(query, db, index_path, k=RETRIEVAL_K)
        
        # Apply manual filtering if Chroma filter is too complex
        if filters:
            candidates = [d for d in candidates if all(d.metadata.get(k) == v for k, v in filters.items())]
            
    except Exception as e:
        logger.warning(f"Hybrid search failed ({e}), falling back to standard search")
        candidates = await asyncio.to_thread(db.similarity_search, query, k=RETRIEVAL_K, filter=filters)
    
    # Enhancement #6: Reranker (already integrated, making it async-aware)
    return await asyncio.to_thread(rerank, query, candidates, top_n=RERANK_TOP_N)

async def generate_rag_response(query: str, scope: str = "global", user_email: Optional[str] = None, chat_history: List[dict] = None) -> dict:
    """Enhanced RAG Response with Context-Awareness and Strict Source Control."""
    
    # Enhancement #8: Query Cache (History-Aware)
    history_hash = hashlib.md5(str(chat_history).encode()).hexdigest() if chat_history else "no_history"
    cache_key = f"{scope}:{user_email}:{query}:{history_hash}"
    
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
    ranked_results = await retrieve_context(query, scope, user_email)
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
            sources.append({"filename": meta.get("filename"), "page": meta.get("page_number")})
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

    try:
        # Use asyncio.to_thread for the blocking Google GenAI call (if no async version used)
        # Actually, genai.Client is sync, so we use to_thread
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=LLM_MODEL,
            config={"system_instruction": system_prompt},
            contents=f"Question: {query}"
        )
        answer = response.text
        
        result = {
            "answer": answer,
            "sources": sources if asks_for_sources else [],
            "model": LLM_MODEL,
            "cached": False
        }
        
        # Save to cache
        _query_cache[cache_key] = result
        await asyncio.to_thread(save_cache)
        
        return result
        
    except Exception as e:
        logger.error(f"OpenRouter error: {e}")
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