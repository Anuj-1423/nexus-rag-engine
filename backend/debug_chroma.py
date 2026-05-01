from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os

BASE_CHROMA_PATH = "storage/chroma/global"

def debug_chunks():
    emb = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory=BASE_CHROMA_PATH, embedding_function=emb)
    
    # Get chunks for Anuj
    res = db.get(where={"filename": "Anuj Singh Resume.pdf"})
    print(f"Total chunks for Anuj Singh Resume.pdf: {len(res['documents'])}")
    for i, doc in enumerate(res['documents']):
        print(f"\n--- Chunk {i} ---")
        print(doc[:500])

if __name__ == "__main__":
    debug_chunks()
