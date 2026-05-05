import requests
import time

BASE_URL = "http://localhost:8000"
USER_EMAIL = "anujsingh@gmail.com"

def ask(question, mode):
    print(f"\n--- Testing Mode: {mode.upper()} ---")
    print(f"Question: {question}")
    
    payload = {
        "email": USER_EMAIL,
        "question": question,
        "mode": mode
    }
    
    try:
        response = requests.post(f"{BASE_URL}/ask", json=payload)
        data = response.json()
        if "detail" in data:
            print(f"Error: {data['detail']}")
        else:
            print(f"Answer: {data.get('answer', 'NO ANSWER')}")
            sources = data.get('sources', [])
            print(f"Sources Found: {len(sources)}")
            for s in sources:
                print(f" - [{s.get('scope', 'unknown')}] {s.get('filename', 'Untitled')} (Page {s.get('page', 1)})")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Wait for rate limit cool down
    print("Waiting 10 seconds for rate limit cool down...")
    time.sleep(10)

    # 1. Enterprise Mode
    ask("What is the company policy? Please mention sources.", "enterprise")
    
    time.sleep(5)
    
    # 2. Personal Mode
    ask("What is my secret vault code? Mention sources.", "personal")
    
    time.sleep(5)
    
    # 3. Combined Mode
    ask("What is the company policy and my secret code? Mention sources.", "combined")
