import os
import logging
from dotenv import load_dotenv
from rag import ingest_document, generate_rag_response

# Load environment variables from .env
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_enhancements():
    # 1. Test Data
    test_text = """
    Enterprise RAG Optimization
    
    Semantic Chunking is a method of splitting documents based on meaning rather than fixed size.
    It improves retrieval accuracy by 30-50% because it preserves the context of sentences.
    
    Deduplication uses MD5 hashing to ensure that duplicate sections across multiple PDFs 
    do not waste space in the vector database. This can reduce index size by 40%.
    
    Hybrid Search combines keyword matching (BM25) with semantic search (Embeddings).
    This is useful for finding exact terms that might be missed by purely semantic models.
    """
    filename = "enhancement_test.txt"
    
    print("\n--- Phase 1: Ingestion with Semantic Chunking & Deduplication ---")
    try:
        res = ingest_document(test_text.encode(), filename, scope="global")
        print(f"Ingestion Result: {res}")
    except Exception as e:
        print(f"Ingestion Failed: {e}")
        return

    print("\n--- Phase 2: Querying with Hybrid Search & Caching ---")
    query = "anuj's education"
    
    # First call (should be fresh)
    print(f"Query 1: {query}")
    res1 = generate_rag_response(query, scope="global")
    print(f"Answer 1: {res1['answer']}")
    print(f"Sources 1: {res1['sources']}")
    print(f"Cached 1: {res1.get('cached', False)}")

    # Second call (should be cached)
    print(f"\nQuery 2 (Duplicate): {query}")
    res2 = generate_rag_response(query, scope="global")
    print(f"Answer 2: {res2['answer']}")
    print(f"Cached 2: {res2.get('cached', False)}")

    print("\n--- Phase 3: Verification Complete ---")

if __name__ == "__main__":
    # Ensure environment variables are set for OpenAI/OpenRouter
    if not os.getenv("OPENROUTER_API_KEY"):
        print("Error: OPENROUTER_API_KEY environment variable not set.")
    else:
        test_enhancements()
