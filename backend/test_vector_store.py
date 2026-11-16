import os
import tempfile
import unittest
from unittest.mock import patch

from langchain_core.documents import Document

from vector_store import QdrantVectorStore


class StubEmbeddings:
    def embed_documents(self, texts):
        return [
            [1.0, 0.0] if "alpha" in text.lower() else [0.0, 1.0]
            for text in texts
        ]

    def embed_query(self, text):
        return [1.0, 0.0] if "alpha" in text.lower() else [0.0, 1.0]


class QdrantVectorStoreTests(unittest.TestCase):
    def test_local_store_search_isolation_and_delete(self):
        with tempfile.TemporaryDirectory() as path:
            embeddings = StubEmbeddings()
            employee_a = QdrantVectorStore(
                collection_name="employee-a",
                embedding_function=embeddings,
                local_path=os.path.join(path, "employee-a", "vectors.sqlite3"),
            )
            employee_b = QdrantVectorStore(
                collection_name="employee-b",
                embedding_function=embeddings,
                local_path=os.path.join(path, "employee-b", "vectors.sqlite3"),
            )

            employee_a.add_documents([
                Document(
                    page_content="Employee A alpha secret",
                    metadata={"filename": "a.txt", "scope": "personal"},
                )
            ])
            employee_b.add_documents([
                Document(
                    page_content="Employee B beta secret",
                    metadata={"filename": "b.txt", "scope": "personal"},
                )
            ])

            self.assertEqual(employee_a.count(), 1)
            self.assertEqual(employee_b.count(), 1)
            self.assertEqual(
                employee_a.similarity_search("alpha", k=1)[0].metadata["filename"],
                "a.txt",
            )
            self.assertEqual(
                employee_b.similarity_search("beta", k=1)[0].metadata["filename"],
                "b.txt",
            )
            self.assertEqual(employee_a.get(where={"filename": "b.txt"})["ids"], [])

            employee_a.delete_where({"filename": "a.txt"})
            self.assertEqual(employee_a.count(), 0)
            self.assertEqual(employee_b.count(), 1)

    def test_cloud_calls_use_qdrant_rest_api(self):
        store = QdrantVectorStore(
            collection_name="nexus-global",
            embedding_function=StubEmbeddings(),
            url="https://qdrant.example",
            api_key="test-key",
        )
        response = {
            "result": {
                "points": [
                    {
                        "id": "point-1",
                        "score": 0.9,
                        "payload": {
                            "page_content": "alpha document",
                            "metadata": {"filename": "a.txt"},
                        },
                    }
                ]
            }
        }
        with patch.object(store, "collection_exists", return_value=True), patch.object(
            store, "_request", side_effect=[{}, response, {"result": {"count": 1}}, {}, {}]
        ) as request:
            store.add_documents([
                Document(page_content="alpha document", metadata={"filename": "a.txt"})
            ])
            matches = store.similarity_search_with_score("alpha", k=1)
            self.assertEqual(matches[0][0].metadata["filename"], "a.txt")
            self.assertAlmostEqual(matches[0][1], 0.1)
            self.assertEqual(store.count(), 1)
            store.delete(["point-1"])
            store.delete_where({"filename": "a.txt"})

        calls = request.call_args_list
        self.assertEqual(calls[0].args[:2], ("PUT", "/collections/nexus-global/points"))
        self.assertEqual(calls[1].args[:2], ("POST", "/collections/nexus-global/points/query"))
        self.assertEqual(calls[2].args[:2], ("POST", "/collections/nexus-global/points/count"))
        self.assertEqual(calls[3].kwargs["json"], {"points": ["point-1"]})
        self.assertEqual(
            calls[4].kwargs["json"],
            {
                "filter": {
                    "must": [
                        {
                            "key": "metadata.filename",
                            "match": {"value": "a.txt"},
                        }
                    ]
                }
            },
        )


if __name__ == "__main__":
    unittest.main()
