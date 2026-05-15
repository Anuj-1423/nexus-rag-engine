
import asyncio
import os
from rag import ingest_document

def simulate_personal_upload():
    content = b"Anuj Singh's personal notes: PRD for the new RAG engine. Features: persistence, mode toggling, admin dashboard."
    filename = "Product Requirements Document.txt"
    email = "anujsingh@gmail.com"
    
    print(f"Simulating personal upload of {filename} for {email}...")
    result = ingest_document(content, filename, scope="personal", user_email=email)
    print(f"Ingestion result: {result['status']}")

if __name__ == "__main__":
    simulate_personal_upload()
