
import os
import mysql.connector
import hashlib

# Configuration
DB_USER = "root"
DB_PASSWORD = "Ishan@1423"
DB_NAME = "defaultdb"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_CHROMA_PATH = os.path.join(BASE_DIR, "storage", "chroma")

def get_index_path(scope, user_email=None):
    if scope == "global":
        return os.path.join(BASE_CHROMA_PATH, "global")
    email_hash = hashlib.md5(user_email.lower().encode()).hexdigest()
    return os.path.join(BASE_CHROMA_PATH, "users", email_hash)

def sync_status():
    try:
        conn = mysql.connector.connect(host="localhost", user=DB_USER, password=DB_PASSWORD, database=DB_NAME)
        cursor = conn.cursor()
        
        cursor.execute("SELECT id, filename, owner_email, scope, status FROM documents WHERE status = 'ready'")
        docs = cursor.fetchall()
        
        print(f"Checking {len(docs)} 'ready' documents...")
        
        for doc_id, filename, email, scope, status in docs:
            path = get_index_path(scope, email)
            # Check if directory exists and has chroma data
            exists = os.path.exists(path) and any(f.endswith('.sqlite3') for f in os.listdir(path)) if os.path.exists(path) else False
            
            if not exists:
                print(f"FIX: Document {doc_id} ('{filename}') is marked 'ready' but vector data is MISSING at {path}. Marking as 'failed'.")
                cursor.execute("UPDATE documents SET status = 'failed', error_message = 'Vector data missing. Please re-upload.' WHERE id = %s", (doc_id,))
            else:
                print(f"OK: Document {doc_id} ('{filename}') has vector data at {path}.")
                
        conn.commit()
        cursor.close()
        conn.close()
        print("Sync complete.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    sync_status()
