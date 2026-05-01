import sys
import os
import asyncio
import time
from dotenv import load_dotenv

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag import generate_rag_response, ingest_document, get_embeddings

load_dotenv(override=True)

async def test_performance_with_data():
    print("--- Performance Test with Data ---")
    
    # 1. Ingest sample data to ensure something is there
    print("Ingesting sample data...")
    sample_text = "The company's CEO is Jane Doe. She joined in 2020. The office is in San Francisco."
    ingest_document(sample_text.encode(), "sample_perf.txt", scope="global")
    
    # 2. Warm up models
    print("Warming up models...")
    await asyncio.to_thread(get_embeddings)

    # 3. Query 1: Should build BM25 cache
    q1 = "Who is the CEO?"
    print(f"\nQuery 1 (Building BM25 Cache): {q1}")
    start = time.time()
    res1 = await generate_rag_response(q1, scope="global")
    print(f"Time: {time.time() - start:.2f}s")
    print(f"Answer: {res1['answer']}")

    # 4. Query 2: Should use BM25 cache
    q2 = "When did she join?"
    print(f"\nQuery 2 (Using BM25 Cache): {q2}")
    start = time.time()
    res2 = await generate_rag_response(q2, scope="global")
    print(f"Time: {time.time() - start:.2f}s")
    print(f"Answer: {res2['answer']}")

if __name__ == "__main__":
    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY not found in environment.")
        sys.exit(1)
        
    asyncio.run(test_performance_with_data())
