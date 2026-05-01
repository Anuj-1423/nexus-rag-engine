import sys
import os

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag import generate_rag_response
from dotenv import load_dotenv

load_dotenv(override=True)

def test_conditional():
    # Case 1: Simple Question
    q1 = "What is the capital of France?"
    print(f"\nQUERY 1: {q1}")
    res1 = generate_rag_response(q1, scope="global")
    print(f"ANSWER: {res1['answer']}")
    print(f"SOURCES IN JSON: {res1['sources']}")

    # Case 2: Question asking for page
    q2 = "On which page is the information about the Eiffel Tower situated?"
    print(f"\nQUERY 2: {q2}")
    res2 = generate_rag_response(q2, scope="global")
    print(f"ANSWER: {res2['answer']}")
    print(f"SOURCES IN JSON: {res2['sources']}")

if __name__ == "__main__":
    test_conditional()
