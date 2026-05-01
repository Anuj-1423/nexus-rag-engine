import os
import hashlib
from rag import ingest_document, get_index_path

def test_personal_brain_isolation():
    # User 1
    user1 = "user1@example.com"
    user1_file = b"This is user 1 private data."
    user1_name = "secret1.txt"
    
    # User 2
    user2 = "user2@example.com"
    user2_file = b"This is user 2 private data."
    user2_name = "secret2.txt"
    
    print(f"--- Testing Personal Brain Isolation ---")
    
    # Ingest for User 1
    print(f"Ingesting for {user1}...")
    ingest_document(user1_file, user1_name, scope="personal", user_email=user1)
    path1 = get_index_path(scope="personal", user_email=user1)
    
    # Ingest for User 2
    print(f"Ingesting for {user2}...")
    ingest_document(user2_file, user2_name, scope="personal", user_email=user2)
    path2 = get_index_path(scope="personal", user_email=user2)
    
    print(f"Path 1: {path1}")
    print(f"Path 2: {path2}")
    
    if path1 != path2 and os.path.exists(path1) and os.path.exists(path2):
        print("✅ SUCCESS: Different users have different vector DB directories.")
    else:
        print("❌ FAILED: Directories are the same or missing.")

if __name__ == "__main__":
    test_personal_brain_isolation()
