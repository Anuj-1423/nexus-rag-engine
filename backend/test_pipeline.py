import os
from dotenv import load_dotenv
from rag import ingest_document, generate_rag_response

def run_full_pipeline_test():
    load_dotenv(override=True)
    
    print("\n--- Starting Full 100% FREE RAG Pipeline Test ---")
    
    # 1. Ingestion Test
    test_file = "test_doc.txt"
    if not os.path.exists(test_file):
        with open(test_file, "w") as f:
            f.write("The Capital of France is Paris. The Eiffel Tower was built in 1889.")
            
    with open(test_file, "rb") as f:
        file_bytes = f.read()
    
    print(f"[STEP 1] Ingesting '{test_file}' using LOCAL Embeddings...")
    try:
        ingest_result = ingest_document(file_bytes, test_file, scope="global")
        print(f"[SUCCESS] Ingestion complete: {ingest_result}")
    except Exception as e:
        print(f"[FAILED] Ingestion error: {e}")
        return

    # 2. Vector DB Check
    index_path = "storage/vectors/global"
    print(f"[STEP 2] Verifying Vector DB on disk at '{index_path}'...")
    if os.path.exists(index_path):
        files = os.listdir(index_path)
        print(f"[SUCCESS] Vector DB found on disk. Files: {files}")
    else:
        print("[FAILED] Vector DB files NOT found.")
        return

    # 3. Retrieval/Query Test
    question = "What is the capital of France and when was the Eiffel Tower built?"
    print(f"[STEP 3] Querying OpenRouter (Llama 3.3 70B): '{question}'...")
    try:
        response = generate_rag_response(question, scope="global")
        print(f"[SUCCESS] AI Response: {response['answer']}")
        print(f"[INFO] Sources used: {[s['filename'] for s in response['sources']]}")
    except Exception as e:
        print(f"[FAILED] Query failed: {e}")

if __name__ == "__main__":
    run_full_pipeline_test()
