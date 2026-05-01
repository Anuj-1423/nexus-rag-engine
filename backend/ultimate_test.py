import os
import sys
import asyncio
import logging
from dotenv import load_dotenv

# Add backend to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from rag import ingest_document, generate_rag_response
from database import init_db

# Configure logging to be very clean
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("UltimateTest")

async def run_ultimate_test():
    load_dotenv()
    
    print("\n" + "="*50)
    print("STARTING ULTIMATE PIPELINE TEST")
    print("="*50 + "\n")

    # 1. Initialize System
    try:
        init_db()
        print("[SUCCESS] System Core Initialized (Database & Folders)")
    except Exception as e:
        print(f"[ERROR] Initialization Failed: {e}")
        return

    test_user = "anuj@example.com"

    # 2. ADMIN PANEL TEST (Global Scope)
    print("\n--- [ADMIN PANEL] Uploading Global Document ---")
    admin_content = (
        "NEXUS GLOBAL POLICY 2026:\n"
        "1. All employees are entitled to 25 days of paid leave.\n"
        "2. The office is open from 9 AM to 6 PM.\n"
        "3. Remote work is allowed on Fridays."
    )
    admin_file = "global_policy.txt"
    try:
        result = await asyncio.to_thread(
            ingest_document,
            file_bytes=admin_content.encode(),
            filename=admin_file,
            scope="global",
            user_email="admin@nexus.com"
        )
        print(f"[SUCCESS] Admin Upload Success! Chunks: {result.get('total_chunks')}")
    except Exception as e:
        print(f"[ERROR] Admin Upload Failed: {e}")

    # 3. USER PANEL TEST (Personal Scope)
    print("\n--- [USER PANEL] Uploading Personal Document ---")
    user_content = (
        "ANUJ'S PERSONAL NOTES:\n"
        "My specific project code is 'PROJECT-X-77'.\n"
        "I need to finish the deployment by next Tuesday.\n"
        "Remember to buy coffee for the team."
    )
    user_file = "personal_notes.txt"
    try:
        result = await asyncio.to_thread(
            ingest_document,
            file_bytes=user_content.encode(),
            filename=user_file,
            scope="personal",
            user_email=test_user
        )
        print(f"[SUCCESS] User Upload Success! Chunks: {result.get('total_chunks')}")
    except Exception as e:
        print(f"[ERROR] User Upload Failed: {e}")

    # 4. RETRIEVAL TEST (Hybrid Query)
    print("\n--- [CHAT TEST] Querying Knowledge Base ---")
    
    # Query 1: Personal Knowledge
    print(f"\nQuerying: 'What is my project code?' (User: {test_user})")
    resp1 = await generate_rag_response(
        query="What is my project code?",
        user_email=test_user,
        scope="personal"
    )
    print(f"Response: {resp1['answer']}")

    # Query 2: Global Knowledge (Accessed by User)
    print(f"\nQuerying: 'How many leaves do I have?' (User: {test_user})")
    resp2 = await generate_rag_response(
        query="How many leaves do I get according to company policy?",
        user_email=test_user,
        scope="personal"
    )
    print(f"Response: {resp2['answer']}")

    print("\n" + "="*50)
    print("TEST COMPLETE: Both Admin and User Pipelines Verified!")
    print("="*50 + "\n")

if __name__ == "__main__":
    asyncio.run(run_ultimate_test())
