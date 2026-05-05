import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_isolation():
    print("--- Starting Isolation Verification ---")
    
    # 1. Register two users
    user_a = {
        "full_name": "User Alpha",
        "email": f"alpha_{int(time.time())}@test.com",
        "password": "password123",
        "role": "employee"
    }
    user_b = {
        "full_name": "User Beta",
        "email": f"beta_{int(time.time())}@test.com",
        "password": "password123",
        "role": "employee"
    }

    print(f"Registering User A: {user_a['email']}")
    requests.post(f"{BASE_URL}/register", json=user_a)
    
    print(f"Registering User B: {user_b['email']}")
    requests.post(f"{BASE_URL}/register", json=user_b)

    # 2. Login User A
    print("Logging in User A...")
    login_a = requests.post(f"{BASE_URL}/login", json={"email": user_a["email"], "password": user_a["password"]}).json()
    token_a = login_a["token"]

    # 3. User A uploads a document
    print("User A uploading a document...")
    files = {'file': ('test_alpha.txt', 'This is content for Alpha.')}
    requests.post(f"{BASE_URL}/upload?email={user_a['email']}&scope=personal", 
                  headers={"Authorization": f"Bearer {token_a}"}, 
                  files=files)

    # 4. User A chats
    print("User A sending a chat...")
    requests.post(f"{BASE_URL}/ask", 
                  headers={"Authorization": f"Bearer {token_a}"}, 
                  json={"email": user_a["email"], "question": "Who am I?", "mode": "standard"})

    # 5. Login User B
    print("Logging in User B...")
    login_b = requests.post(f"{BASE_URL}/login", json={"email": user_b["email"], "password": user_b["password"]}).json()
    token_b = login_b["token"]

    # 6. Verify User B cannot see User A's documents
    print("Verifying User B document isolation...")
    docs_b = requests.get(f"{BASE_URL}/documents/{user_b['email']}?scope=personal", 
                          headers={"Authorization": f"Bearer {token_b}"}).json()
    
    # Check if User A's doc is in User B's list
    found_alpha_doc = any(d['filename'] == 'test_alpha.txt' for d in docs_b)
    if found_alpha_doc:
        print("FAIL: User B can see User A's document!")
    else:
        print("SUCCESS: User B cannot see User A's documents.")

    # 7. Verify User B cannot see User A's chat history
    print("Verifying User B chat isolation...")
    history_b = requests.get(f"{BASE_URL}/history/{user_b['email']}?scope=personal", 
                             headers={"Authorization": f"Bearer {token_b}"}).json()
    
    found_alpha_chat = any("Who am I?" in c['question'] for c in history_b)
    if found_alpha_chat:
        print("FAIL: User B can see User A's chat history!")
    else:
        print("SUCCESS: User B cannot see User A's chat history.")

    # 8. Malicious Attempt: User B tries to access User A's docs via path tampering
    print("Malicious Attempt: User B tries to access User A's library endpoint...")
    malicious_res = requests.get(f"{BASE_URL}/documents/{user_a['email']}?scope=personal", 
                                 headers={"Authorization": f"Bearer {token_b}"})
    
    if malicious_res.status_code == 403:
        print("SUCCESS: Server blocked User B from accessing User A's document endpoint (403 Forbidden).")
    else:
        print(f"FAIL: Server allowed access or returned {malicious_res.status_code}.")

    # 9. Malicious Attempt: User B tries to delete User A's document
    print("Malicious Attempt: User B tries to delete User A's document...")
    delete_res = requests.delete(f"{BASE_URL}/documents/{user_a['email']}/test_alpha.txt?scope=personal", 
                                 headers={"Authorization": f"Bearer {token_b}"})
    
    if delete_res.status_code == 403:
        print("SUCCESS: Server blocked User B from deleting User A's document (403 Forbidden).")
    else:
        print(f"FAIL: Server allowed deletion or returned {delete_res.status_code}.")

    print("--- Verification Complete ---")

if __name__ == "__main__":
    test_isolation()
