import json
import math
import os
import sqlite3
import uuid
from contextlib import contextmanager
from typing import Optional
from urllib.parse import quote

import requests
from langchain_core.documents import Document


class QdrantVectorStore:
    def __init__(
        self,
        collection_name: str,
        embedding_function,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        local_path: Optional[str] = None,
    ):
        self.collection_name = collection_name
        self.embedding_function = embedding_function
        self.url = url.rstrip("/") if url else None
        self.local_path = local_path
        if self.url:
            if not api_key:
                raise ValueError("QDRANT_API_KEY is required when QDRANT_URL is configured.")
            self.session = requests.Session()
            self.session.headers.update({"api-key": api_key})
        else:
            if not local_path:
                raise ValueError("local_path is required when QDRANT_URL is not configured.")
            self.session = None

    @property
    def _collection_url(self) -> str:
        return f"/collections/{quote(self.collection_name, safe='')}"

    def _request(self, method: str, path: str, **kwargs) -> dict:
        response = self.session.request(
            method,
            f"{self.url}{path}",
            timeout=30,
            **kwargs,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as error:
            detail = response.text[:1000]
            raise requests.HTTPError(
                f"{error}. Qdrant response: {detail}",
                response=response,
            ) from error
        return response.json()

    def _connect_local(self) -> sqlite3.Connection:
        parent = os.path.dirname(self.local_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        return sqlite3.connect(self.local_path, timeout=30)

    @contextmanager
    def _local_connection(self):
        connection = self._connect_local()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _ensure_local_collection(self) -> None:
        with self._local_connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS vectors (
                    id TEXT PRIMARY KEY,
                    vector TEXT NOT NULL,
                    page_content TEXT NOT NULL,
                    metadata TEXT NOT NULL
                )
                """
            )

    def collection_exists(self) -> bool:
        if not self.url:
            with self._local_connection() as connection:
                row = connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'vectors'"
                ).fetchone()
                return row is not None

        response = self.session.get(
            f"{self.url}{self._collection_url}",
            timeout=30,
        )
        if response.status_code == 404:
            return False
        response.raise_for_status()
        return True

    def get_vector_size(self) -> int:
        if not self.url:
            raise ValueError("Vector dimensions are only available from Qdrant Cloud collections.")
        collection = self._request("GET", self._collection_url).get("result") or {}
        vectors = (collection.get("config") or {}).get("params", {}).get("vectors")
        if not isinstance(vectors, dict) or not isinstance(vectors.get("size"), int):
            raise ValueError(
                f"Collection '{self.collection_name}' must use a single unnamed vector."
            )
        return vectors["size"]

    def _ensure_payload_indexes(self, where: Optional[dict]) -> None:
        if not self.url or not where:
            return

        collection = self._request("GET", self._collection_url).get("result") or {}
        payload_schema = collection.get("payload_schema") or {}
        for key in where:
            field_name = f"metadata.{key}"
            if field_name in payload_schema:
                continue
            try:
                self._request(
                    "PUT",
                    f"{self._collection_url}/index",
                    params={"wait": "true"},
                    json={"field_name": field_name, "field_schema": "keyword"},
                )
            except requests.HTTPError as error:
                if error.response is None or error.response.status_code != 409:
                    raise
                refreshed = self._request("GET", self._collection_url).get("result") or {}
                if field_name not in (refreshed.get("payload_schema") or {}):
                    raise

    def _ensure_collection(self, vector_size: int) -> None:
        if not self.url:
            self._ensure_local_collection()
            return
        if self.collection_exists():
            return
        try:
            self._request(
                "PUT",
                self._collection_url,
                json={
                    "vectors": {
                        "size": vector_size,
                        "distance": "Cosine",
                    }
                },
            )
        except requests.HTTPError as error:
            if error.response is None or error.response.status_code != 409:
                raise
            if not self.collection_exists():
                raise

    @staticmethod
    def _build_filter(where: Optional[dict]) -> Optional[dict]:
        if not where:
            return None
        return {
            "must": [
                {"key": f"metadata.{key}", "match": {"value": value}}
                for key, value in where.items()
            ]
        }

    @staticmethod
    def _matches(metadata: dict, where: Optional[dict]) -> bool:
        return not where or all(metadata.get(key) == value for key, value in where.items())

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
        vector_size = len(vectors[0])
        if any(len(vector) != vector_size for vector in vectors):
            raise ValueError("Embedding provider returned vectors with inconsistent dimensions.")
        self._ensure_collection(vector_size)

        for offset in range(0, len(documents), 64):
            batch_documents = documents[offset:offset + 64]
            batch_vectors = vectors[offset:offset + 64]
            points = [
                {
                    "id": str(uuid.uuid4()),
                    "vector": vector,
                    "payload": {
                        "page_content": document.page_content,
                        "metadata": document.metadata,
                    },
                }
                for document, vector in zip(batch_documents, batch_vectors)
            ]
            if self.url:
                self._request(
                    "PUT",
                    f"{self._collection_url}/points",
                    params={"wait": "true"},
                    json={"points": points},
                )
            else:
                with self._local_connection() as connection:
                    connection.executemany(
                        """
                        INSERT OR REPLACE INTO vectors
                            (id, vector, page_content, metadata)
                        VALUES (?, ?, ?, ?)
                        """,
                        [
                            (
                                point["id"],
                                json.dumps(point["vector"]),
                                point["payload"]["page_content"],
                                json.dumps(point["payload"]["metadata"]),
                            )
                            for point in points
                        ],
                    )

    def get(self, where: Optional[dict] = None, limit: Optional[int] = None) -> dict:
        if not self.collection_exists() or limit == 0:
            return {"ids": [], "documents": [], "metadatas": []}
        self._ensure_payload_indexes(where)

        ids = []
        documents = []
        metadatas = []
        offset = None
        remaining = limit
        while True:
            page_limit = min(256, remaining) if remaining is not None else 256
            if self.url:
                response = self._request(
                    "POST",
                    f"{self._collection_url}/points/scroll",
                    json={
                        "filter": self._build_filter(where),
                        "limit": page_limit,
                        "offset": offset,
                        "with_payload": True,
                        "with_vector": False,
                    },
                )
                result = response.get("result") or {}
                points = result.get("points") or []
                offset = result.get("next_page_offset")
            else:
                with self._local_connection() as connection:
                    rows = connection.execute(
                        "SELECT id, page_content, metadata FROM vectors ORDER BY rowid"
                    ).fetchall()
                matching_rows = [
                    row for row in rows
                    if self._matches(json.loads(row[2]), where)
                ]
                if remaining is not None:
                    matching_rows = matching_rows[:remaining]
                points = [
                    {
                        "id": row[0],
                        "payload": {
                            "page_content": row[1],
                            "metadata": json.loads(row[2]),
                        },
                    }
                    for row in matching_rows
                ]
                offset = None

            for point in points:
                payload = point.get("payload") or {}
                ids.append(str(point["id"]))
                documents.append(payload.get("page_content", ""))
                metadatas.append(payload.get("metadata") or {})
            if offset is None or (remaining is not None and len(ids) >= limit):
                break
            if remaining is not None:
                remaining = limit - len(ids)
        return {"ids": ids, "documents": documents, "metadatas": metadatas}

    def similarity_search_with_score(self, query: str, k: int = 4) -> list:
        if not self.collection_exists() or k <= 0:
            return []
        vector = self.embedding_function.embed_query(query)
        if self.url:
            response = self._request(
                "POST",
                f"{self._collection_url}/points/query",
                json={
                    "query": vector,
                    "limit": k,
                    "with_payload": True,
                },
            )
            points = (response.get("result") or {}).get("points") or []
            return [
                (
                    Document(
                        page_content=(point.get("payload") or {}).get("page_content", ""),
                        metadata=(point.get("payload") or {}).get("metadata") or {},
                    ),
                    1.0 - float(point["score"]),
                )
                for point in points
            ]

        with self._local_connection() as connection:
            rows = connection.execute(
                "SELECT vector, page_content, metadata FROM vectors"
            ).fetchall()
        query_norm = math.sqrt(sum(value * value for value in vector))
        scored = []
        for stored_vector, page_content, metadata_json in rows:
            stored = json.loads(stored_vector)
            if len(stored) != len(vector):
                raise ValueError("Query embedding dimension does not match stored vectors.")
            stored_norm = math.sqrt(sum(value * value for value in stored))
            similarity = (
                sum(left * right for left, right in zip(vector, stored)) / (query_norm * stored_norm)
                if query_norm and stored_norm
                else 0.0
            )
            scored.append((
                Document(page_content=page_content, metadata=json.loads(metadata_json)),
                1.0 - similarity,
            ))
        scored.sort(key=lambda result: result[1])
        return scored[:k]

    def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        return [document for document, _ in self.similarity_search_with_score(query, k)]

    def count(self) -> int:
        if not self.collection_exists():
            return 0
        if self.url:
            response = self._request(
                "POST",
                f"{self._collection_url}/points/count",
                json={"exact": True},
            )
            return int((response.get("result") or {}).get("count", 0))
        with self._local_connection() as connection:
            return connection.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]

    def delete(self, ids: list[str]) -> None:
        if not ids:
            return
        if self.url:
            self._request(
                "POST",
                f"{self._collection_url}/points/delete",
                params={"wait": "true"},
                json={"points": ids},
            )
            return
        with self._local_connection() as connection:
            connection.executemany("DELETE FROM vectors WHERE id = ?", [(point_id,) for point_id in ids])

    def delete_where(self, where: dict) -> None:
        if not self.collection_exists():
            return
        if self.url:
            self._ensure_payload_indexes(where)
            self._request(
                "POST",
                f"{self._collection_url}/points/delete",
                params={"wait": "true"},
                json={"filter": self._build_filter(where)},
            )
            return
        with self._local_connection() as connection:
            rows = connection.execute("SELECT id, metadata FROM vectors").fetchall()
            matching_ids = [
                (point_id,)
                for point_id, metadata in rows
                if self._matches(json.loads(metadata), where)
            ]
            connection.executemany("DELETE FROM vectors WHERE id = ?", matching_ids)
