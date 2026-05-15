
import asyncio
import os
from rag import ingest_document

def simulate_admin_upload():
    # We'll use a dummy text to simulate a document
    content = b"This is a global enterprise document about Nexus RAG Engine policy. All employees must follow these rules."
    filename = "Nexus_Policy.txt"
    meta = {
        "filename": filename,
        "email": "sita@gmail.com",
        "scope": "global"
    }
    
    print(f"Simulating global upload of {filename}...")
    result = ingest_document(content, filename, scope="global", user_email="sita@gmail.com")
    print(f"Ingestion result: {result['status']}")
    if "error" in result:
        print(f"Error: {result['error']}")

if __name__ == "__main__":
    simulate_admin_upload()
