
import asyncio
from rag import get_embeddings, get_vector_store, get_index_path

async def debug_db():
    email = "anujsingh@gmail.com"
    path = get_index_path("personal", email)
    print(f"Path: {path}")
        
    emb = get_embeddings()
    db = get_vector_store("personal", email, emb)
    if not db.collection_exists():
        print("Vector collection does not exist.")
        return
    
    data = db.get()
    print(f"Documents count: {len(data['documents'])}")
    if data['documents']:
        print(f"First document snippet: {data['documents'][0][:100]}")
        print(f"Metadatas: {data['metadatas'][0]}")

if __name__ == "__main__":
    asyncio.run(debug_db())
