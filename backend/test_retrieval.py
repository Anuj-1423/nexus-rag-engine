
import asyncio
import os
from rag import retrieve_context

async def test_retrieval():
    email = "anujsingh@gmail.com"
    query = "Product Requirements Document"
    
    print(f"Testing retrieval for {email} with mode='combined'...")
    results = await retrieve_context(query, mode="combined", user_email=email)
    print(f"Results found: {len(results)}")
    for i, doc in enumerate(results):
        # ranked_results return depends on rerank. 
        # Usually it's [(doc, score), ...]
        if isinstance(doc, tuple):
             d, s = doc
             print(f"{i+1}. [{d.metadata.get('filename')}] (Score: {s:.4f})")
        else:
             print(f"{i+1}. [{doc.metadata.get('filename')}]")

if __name__ == "__main__":
    asyncio.run(test_retrieval())
