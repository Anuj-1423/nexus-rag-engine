
import os
import mysql.connector
import hashlib

# Database Configuration
DB_HOST = 'localhost'
DB_USER = 'root'
DB_PASSWORD = 'Ishan@1423'
DB_NAME = 'defaultdb'
DB_PORT = 3306

BASE_DIR = r"c:\Users\Ishaan\Documents\nexus-rag-engine-main"
BASE_CHROMA_PATH = os.path.join(BASE_DIR, "storage", "chroma")

def get_index_path(scope: str = "global", user_email: str = None) -> str:
    if scope == "global":
        return os.path.join(BASE_CHROMA_PATH, "global")
    if not user_email:
        return None
    email_hash = hashlib.md5(user_email.lower().encode()).hexdigest()
    return os.path.join(BASE_CHROMA_PATH, "users", email_hash)

def check_status():
    try:
        conn = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            port=DB_PORT,
            database=DB_NAME
        )
        cursor = conn.cursor()
        
        print("Checking documents table...")
        cursor.execute("SELECT id, filename, status, scope, owner_email FROM documents")
        rows = cursor.fetchall()
        
        print(f"{'ID':<5} | {'Filename':<30} | {'Status':<10} | {'Scope':<10} | {'Owner':<30} | {'Chroma Exists'}")
        print("-" * 110)
        
        for row in rows:
            doc_id, filename, status, scope, owner = row
            path = get_index_path(scope, owner)
            exists = os.path.exists(path) if path else "N/A"
            print(f"{doc_id:<5} | {filename[:30]:<30} | {status:<10} | {scope:<10} | {owner[:30]:<30} | {exists}")
            
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_status()
