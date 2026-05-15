
import mysql.connector
import os
from database import get_db_connection, DB_NAME

def dump_docs():
    conn = get_db_connection(DB_NAME)
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT id, filename, status, scope, owner_email, error_message FROM documents")
        docs = cursor.fetchall()
        print(f"{'ID':<4} | {'Filename':<30} | {'Status':<10} | {'Scope':<10} | {'Owner':<20} | {'Error'}")
        print("-" * 120)
        for doc in docs:
            error = (doc['error_message'] or "")[:50]
            print(f"{doc['id']:<4} | {doc['filename']:<30} | {doc['status']:<10} | {doc['scope']:<10} | {doc['owner_email']:<20} | {error}")
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    dump_docs()
