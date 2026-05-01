import requests
import os
import json
import time

BASE_URL = "http://localhost:8000" # Updated to 8000

def test_multi_mode_retrieval():
    user_email = "test_user@nexus.com"
    
    # 1. Upload Enterprise Doc (Global)
    print("Uploading Enterprise Document...")
    with open("enterprise_info.txt", "w") as f:
        f.write("The company policy for 2026 states that all employees get 30 days of vacation.")
    
    with open("enterprise_info.txt", "rb") as f:
        r = requests.post(f"{BASE_URL}/upload?email=admin@nexus.com&scope=global", files={"file": f})
        print(f"Global Upload Status: {r.status_code}, Response: {r.text}")

    # 2. Upload Personal Doc (User)
    print("\nUploading Personal Document...")
    with open("personal_info.txt", "w") as f:
        f.write("My personal bank account number is 987654321 and my favorite color is Cyan.")
    
    with open("personal_info.txt", "rb") as f:
        r = requests.post(f"{BASE_URL}/upload?email={user_email}&scope=personal", files={"file": f})
        print(f"Personal Upload Status: {r.status_code}, Response: {r.text}")

    time.sleep(2) # Give it a second to index

    # 3. Test ENTERPRISE Mode
    print("\n--- Testing ENTERPRISE Mode ---")
    payload = {"email": user_email, "question": "What is the vacation policy?", "mode": "enterprise"}
    r = requests.post(f"{BASE_URL}/ask", json=payload)
    print(f"Q: Vacation Policy? -> {r.json().get('answer')[:100]}...")

    payload = {"email": user_email, "question": "What is my bank account number?", "mode": "enterprise"}
    r = requests.post(f"{BASE_URL}/ask", json=payload)
    print(f"Q: Bank Account? -> {r.json().get('answer')[:100]}...")

    # 4. Test PERSONAL Mode
    print("\n--- Testing PERSONAL Mode ---")
    payload = {"email": user_email, "question": "What is the vacation policy?", "mode": "personal"}
    r = requests.post(f"{BASE_URL}/ask", json=payload)
    print(f"Q: Vacation Policy? -> {r.json().get('answer')[:100]}...")

    payload = {"email": user_email, "question": "What is my bank account number?", "mode": "personal"}
    r = requests.post(f"{BASE_URL}/ask", json=payload)
    print(f"Q: Bank Account? -> {r.json().get('answer')[:100]}...")

    # 5. Test COMBINED Mode
    print("\n--- Testing COMBINED Mode ---")
    payload = {"email": user_email, "question": "Tell me about vacation policy and my favorite color.", "mode": "combined"}
    r = requests.post(f"{BASE_URL}/ask", json=payload)
    print(f"Q: Vacation & Color? -> {r.json().get('answer')[:200]}...")

if __name__ == "__main__":
    # Clean up old test files if any
    for f in ["enterprise_info.txt", "personal_info.txt"]:
        if os.path.exists(f): os.remove(f)
    
    try:
        test_multi_mode_retrieval()
    except Exception as e:
        print(f"Test failed: {e}")
    finally:
        for f in ["enterprise_info.txt", "personal_info.txt"]:
            if os.path.exists(f): os.remove(f)
