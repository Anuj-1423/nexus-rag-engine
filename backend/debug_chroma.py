
import asyncio
import os
from langchain_chroma import Chroma
from rag import get_embeddings, get_index_path

async def debug_db():
    email = "anujsingh@gmail.com"
    path = get_index_path("personal", email)
    print(f"Path: {path}")
    if not os.path.exists(path):
        print("Path does not exist!")
        return
        
    emb = get_embeddings()
    db = Chroma(persist_directory=path, embedding_function=emb)
    
    data = db.get()
    print(f"Documents count: {len(data['documents'])}")
    if data['documents']:
        print(f"First document snippet: {data['documents'][0][:100]}")
        print(f"Metadatas: {data['metadatas'][0]}")

if __name__ == "__main__":
    asyncio.run(debug_db())
