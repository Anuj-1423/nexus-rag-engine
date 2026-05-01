import os
from dotenv import load_dotenv
load_dotenv()
from rag import ingest_document, generate_rag_response

def test_personal_pipeline():
    user_email = "testuser@example.com"
    filename = "secret_project_x.txt"
    file_content = b"Project X is a highly confidential project that aims to build a time machine using quantum flux capacitors. Only testuser knows about this."

    print(f"--- Testing Personal Pipeline for {user_email} ---")

    # Step 1: Ingest into Personal Brain
    try:
        print("1. Ingesting document into personal brain...")
        result = ingest_document(file_content, filename, scope="personal", user_email=user_email)
        print(f"   Success! Ingest result: {result}")
    except Exception as e:
        print(f"   [FAILED] Ingestion failed: {e}")
        return

    # Step 2: Query Personal Brain
    try:
        print("\n2. Querying personal brain about Project X...")
        query = "What is Project X?"
        response = generate_rag_response(query, scope="personal", user_email=user_email)
        
        answer = response.get("answer", "")
        sources = response.get("sources", [])
        
        print(f"   Query: {query}")
        print(f"   Answer: {answer}")
        print(f"   Sources: {sources}")
        
        if "time machine" in answer.lower() or "quantum" in answer.lower():
            print("\n[SUCCESS] PIPELINE SUCCESS: The personal query successfully retrieved the personal document.")
        else:
            print("\n[FAILED] PIPELINE FAILED: The answer did not contain the expected personal information.")
            
    except Exception as e:
        print(f"   [FAILED] Query failed: {e}")

    # Step 3: Delete from Personal Brain
    try:
        from rag import delete_document_from_vector
        print("\n3. Deleting document from personal brain...")
        success = delete_document_from_vector(filename, scope="personal", user_email=user_email)
        if success:
            print("   [SUCCESS] Deletion successful.")
        else:
            print("   [FAILED] Deletion failed.")
            
        print("\n4. Verifying deletion (should return empty response)...")
        response2 = generate_rag_response(query, scope="personal", user_email=user_email)
        print(f"   Answer after deletion: {response2.get('answer')}")
        if not response2.get('sources'):
            print("   [SUCCESS] Verified: Document is no longer retrievable.")
        else:
            print("   [FAILED] Document is still retrievable!")
            
    except Exception as e:
        print(f"   [FAILED] Deletion test failed: {e}")

if __name__ == "__main__":
    test_personal_pipeline()
