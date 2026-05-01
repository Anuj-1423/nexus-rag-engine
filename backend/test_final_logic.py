import sys
import os

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag import generate_rag_response
from dotenv import load_dotenv

load_dotenv(override=True)

def test_context_and_strict_sources():
    # Simulate first turn
    q1 = "Can you tell me Anuj's school?"
    print(f"\nQUERY 1: {q1}")
    res1 = generate_rag_response(q1, scope="global")
    print(f"ANSWER: {res1['answer']}")
    
    # Simulate second turn with history
    history = [{"question": q1, "answer": res1['answer']}]
    q2 = "On which resource is it mentioned?"
    print(f"\nQUERY 2: {q2}")
    res2 = generate_rag_response(q2, scope="global", chat_history=history)
    print(f"ANSWER: {res2['answer']}")
    print(f"SOURCES IN JSON: {res2['sources']}")

if __name__ == "__main__":
    test_context_and_strict_sources()
