import sys
import os
import asyncio
import time
from dotenv import load_dotenv

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag import generate_rag_response, get_embeddings

load_dotenv(override=True)

async def test_performance():
    print("--- Performance Test ---")
    
    # 1. Warm up models (load singleton)
    print("Warming up models...")
    start_warmup = time.time()
    await asyncio.to_thread(get_embeddings)
    print(f"Warm up took {time.time() - start_warmup:.2f}s")

    # 2. First search (will build BM25 cache)
    q1 = "What is this document about?"
    print(f"\nQuery 1 (Cold cache): {q1}")
    start = time.time()
    res1 = await generate_rag_response(q1, scope="global")
    print(f"Time: {time.time() - start:.2f}s")
    print(f"Answer: {res1['answer'][:100]}...")

    # 3. Second search (Should use BM25 cache)
    q2 = "Who is the owner?"
    print(f"\nQuery 2 (Warm cache): {q2}")
    start = time.time()
    res2 = await generate_rag_response(q2, scope="global")
    print(f"Time: {time.time() - start:.2f}s")
    print(f"Answer: {res2['answer'][:100]}...")

    # 4. Third search with history
    q3 = "Can you summarize our conversation?"
    history = [
        {"question": q1, "answer": res1['answer']},
        {"question": q2, "answer": res2['answer']}
    ]
    print(f"\nQuery 3 (With history): {q3}")
    start = time.time()
    res3 = await generate_rag_response(q3, scope="global", chat_history=history)
    print(f"Time: {time.time() - start:.2f}s")
    print(f"Answer: {res3['answer'][:100]}...")

if __name__ == "__main__":
    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY not found in environment.")
        sys.exit(1)
        
    asyncio.run(test_performance())
