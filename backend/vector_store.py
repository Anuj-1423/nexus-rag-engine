import uuid
from typing import Optional

from langchain_core.documents import Document
from qdrant_client import QdrantClient, models


class QdrantVectorStore:
    def __init__(self, client: QdrantClient, collection_name: str, embedding_function):
        self.client = client
        self.collection_name = collection_name
        self.embedding_function = embedding_function

    def collection_exists(self) -> bool:
        return self.client.collection_exists(self.collection_name)

    def _ensure_collection(self, vector_size: int) -> None:
        if self.collection_exists():
            return
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                ),
            )
        except Exception:
            if not self.collection_exists():
                raise

    def add_documents(self, documents: list[Document]) -> None:
        if not documents:
            return
        vectors = self.embedding_function.embed_documents(
            [document.page_content for document in documents]
        )
        if len(vectors) != len(documents):
            raise ValueError("Embedding provider returned a different number of vectors than documents.")
        if not vectors or not vectors[0]:
            raise ValueError("Embedding provider returned an empty vector.")
        self._ensure_collection(len(vectors[0]))

        for offset in range(0, len(documents), 64):
            batch_documents = documents[offset:offset + 64]
            batch_vectors = vectors[offset:offset + 64]
            points = [
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "page_content": document.page_content,
                        "metadata": document.metadata,
                    },
                )
                for document, vector in zip(batch_documents, batch_vectors)
            ]
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )

    def _build_filter(self, where: Optional[dict]) -> Optional[models.Filter]:
        if not where:
            return None
        conditions = [
            models.FieldCondition(
                key=f"metadata.{key}",
                match=models.MatchValue(value=value),
            )
            for key, value in where.items()
        ]
        return models.Filter(must=conditions)

    def get(self, where: Optional[dict] = None, limit: Optional[int] = None) -> dict:
        if not self.collection_exists():
            return {"ids": [], "documents": [], "metadatas": []}

        ids = []
        documents = []
        metadatas = []
        offset = None
        remaining = limit
        while True:
            page_limit = min(256, remaining) if remaining is not None else 256
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=self._build_filter(where),
                limit=page_limit,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                payload = point.payload or {}
                ids.append(str(point.id))
                documents.append(payload.get("page_content", ""))
                metadatas.append(payload.get("metadata") or {})
            if offset is None or (remaining is not None and len(ids) >= limit):
                break
            if remaining is not None:
                remaining = limit - len(ids)
        return {"ids": ids, "documents": documents, "metadatas": metadatas}

    def similarity_search_with_score(self, query: str, k: int = 4) -> list:
        if not self.collection_exists():
            return []
        vector = self.embedding_function.embed_query(query)
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=k,
            with_payload=True,
        )
        results = []
        for point in response.points:
            payload = point.payload or {}
            results.append((
                Document(
                    page_content=payload.get("page_content", ""),
                    metadata=payload.get("metadata") or {},
                ),
                1.0 - float(point.score),
            ))
        return results

    def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        return [document for document, _ in self.similarity_search_with_score(query, k)]

    def count(self) -> int:
        if not self.collection_exists():
            return 0
        return self.client.count(
            collection_name=self.collection_name,
            exact=True,
        ).count

    def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(points=ids),
            wait=True,
        )

    def delete_where(self, where: dict) -> None:
        if not self.collection_exists():
            return
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=self._build_filter(where),
            ),
            wait=True,
        )
