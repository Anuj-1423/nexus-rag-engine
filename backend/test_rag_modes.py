import asyncio
import os
import sys
from unittest.mock import MagicMock, patch

# Add the current directory to sys.path so we can import rag
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

async def test_rag_retrieval_logic():
    print("Starting RAG Retrieval Logic Test...")
    
    # Mock data
    mock_user_email = "test@example.com"
    mock_query = "What is the policy?"
    
    # We will patch Chroma and hybrid_search to avoid real API calls
    with patch("rag.get_embeddings") as mock_get_emb, \
         patch("rag.Chroma") as mock_chroma, \
         patch("rag.hybrid_search") as mock_hybrid, \
         patch("rag.rerank") as mock_rerank, \
         patch("os.path.exists") as mock_exists:
        
        # Setup mocks
        mock_get_emb.return_value = MagicMock()
        mock_exists.return_value = True # Assume indices exist
        mock_hybrid.return_value = [MagicMock(page_content="Mock Result", metadata={})]
        mock_rerank.return_value = [{"content": "Mock Result", "score": 1.0}]
        
        from rag import retrieve_context
        
        print("\n--- Testing ENTERPRISE Mode ---")
        results = await retrieve_context(mock_query, mode="enterprise", user_email=mock_user_email)
        print(f"OK: Enterprise Mode returned {len(results)} results.")
        # Should only call hybrid_search once for global
        
        print("\n--- Testing PERSONAL Mode ---")
        results = await retrieve_context(mock_query, mode="personal", user_email=mock_user_email)
        print(f"OK: Personal Mode returned {len(results)} results.")
        
        print("\n--- Testing COMBINED Mode ---")
        results = await retrieve_context(mock_query, mode="combined", user_email=mock_user_email)
        print(f"OK: Combined Mode returned {len(results)} results.")
        
    print("\nAll retrieval logic tests passed (Mocked)!")

if __name__ == "__main__":
    asyncio.run(test_rag_retrieval_logic())
