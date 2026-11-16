import tempfile
import unittest

from langchain_core.documents import Document
from qdrant_client import QdrantClient

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
    def test_collection_search_isolation_and_delete(self):
        with tempfile.TemporaryDirectory() as path:
            client = QdrantClient(path=path)
            embeddings = StubEmbeddings()
            employee_a = QdrantVectorStore(client, "employee-a", embeddings)
            employee_b = QdrantVectorStore(client, "employee-b", embeddings)

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

            client.close()


if __name__ == "__main__":
    unittest.main()
