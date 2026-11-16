
import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

from database import DB_NAME, get_db_connection
from rag import vector_document_exists

def check_status():
    try:
        conn = get_db_connection(DB_NAME)
        cursor = conn.cursor()
        
        print("Checking documents table...")
        cursor.execute("SELECT id, filename, status, scope, owner_email FROM documents")
        rows = cursor.fetchall()
        
        print(f"{'ID':<5} | {'Filename':<30} | {'Status':<10} | {'Scope':<10} | {'Owner':<30} | {'Vector Exists'}")
        print("-" * 110)
        
        for row in rows:
            doc_id, filename, status, scope, owner = row
            exists = vector_document_exists(filename, scope, owner)
            print(f"{doc_id:<5} | {filename[:30]:<30} | {status:<10} | {scope:<10} | {owner[:30]:<30} | {exists}")
            
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_status()
