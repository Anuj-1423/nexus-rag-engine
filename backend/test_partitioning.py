import asyncio
import os
import sys
from dotenv import load_dotenv

# Load env vars
load_dotenv(".env")

async def test_partitioning():
    from rag import retrieve_context
    
    user_email = "anujsingh@gmail.com"
    
    print("--- Testing Partitioning (Retrieval Only) ---")
    
    # 1. Enterprise Only
    print("\n[Enterprise Mode]")
    res = await retrieve_context("policy coffee leave", mode="enterprise", user_email=user_email)
    for doc_tuple in res:
        doc = doc_tuple[0]
        print(f"OK Found: {doc.metadata.get('filename')} (Scope: {doc.metadata.get('scope')})")
            
    # 2. Personal Only
    print("\n[Personal Mode]")
    res = await retrieve_context("secret vault code Pizza", mode="personal", user_email=user_email)
    for doc_tuple in res:
        doc = doc_tuple[0]
        print(f"OK Found: {doc.metadata.get('filename')} (Scope: {doc.metadata.get('scope')})")

    # 3. Combined Mode
    print("\n[Combined Mode]")
    res = await retrieve_context("policy secret Pizza", mode="combined", user_email=user_email)
    found_scopes = set()
    for doc_tuple in res:
        doc = doc_tuple[0]
        found_scopes.add(doc.metadata.get('scope'))
        print(f"OK Found: {doc.metadata.get('filename')} (Scope: {doc.metadata.get('scope')})")
    
    if "global" in found_scopes and "personal" in found_scopes:
        print("\nSUCCESS: Combined mode correctly retrieved from BOTH scopes!")
    else:
        print(f"\nPartitioning Results: {found_scopes}")

if __name__ == "__main__":
    asyncio.run(test_partitioning())
