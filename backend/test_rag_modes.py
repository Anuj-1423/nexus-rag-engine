import asyncio
import os
import sys
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

# Add the current directory to sys.path so we can import rag
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

async def test_rag_retrieval_logic():
    mock_user_email = "test@example.com"
    mock_query = "What is the policy?"
    from rag import retrieve_context

    class Store:
        def __init__(self, scope):
            self.scope = scope

        def collection_exists(self):
            return True

    scopes_searched = []
    global_doc = Document(page_content="Global company policy", metadata={"scope": "global"})
    personal_doc = Document(page_content="Personal vacation notes", metadata={"scope": "personal"})

    def make_store(scope, user_email, embeddings):
        scopes_searched.append((scope, user_email))
        return Store(scope)

    async def search_scope(query, store, index_path, k):
        return [global_doc] if store.scope == "global" else [personal_doc]

    def rerank(query, docs, top_n):
        return [(doc, 0.1) for doc in docs[:top_n]]

    with patch("rag.get_embeddings", return_value=MagicMock()), \
         patch("rag.get_vector_store", side_effect=make_store), \
         patch("rag.hybrid_search", side_effect=search_scope), \
         patch("rag.rerank", side_effect=rerank):
        enterprise = await retrieve_context(mock_query, mode="enterprise", user_email=mock_user_email)
        assert [scope for scope, _ in scopes_searched] == ["global"]
        assert {doc.metadata["scope"] for doc, _ in enterprise} == {"global"}

        scopes_searched.clear()
        personal = await retrieve_context(mock_query, mode="personal", user_email=mock_user_email)
        assert scopes_searched == [("personal", mock_user_email)]
        assert {doc.metadata["scope"] for doc, _ in personal} == {"personal"}

        scopes_searched.clear()
        combined = await retrieve_context(mock_query, mode="combined", user_email=mock_user_email)
        assert set(scope for scope, _ in scopes_searched) == {"global", "personal"}
        assert any(scope == "personal" and email == mock_user_email for scope, email in scopes_searched)
        assert {doc.metadata["scope"] for doc, _ in combined} == {"global", "personal"}

        scopes_searched.clear()
        owner_emails = ["first@example.com", "second@example.com"]
        admin_combined = await retrieve_context(
            mock_query,
            mode="combined",
            user_email="admin@example.com",
            is_admin=True,
            personal_emails=owner_emails,
        )
        assert set(scopes_searched) == {
            ("global", None),
            ("personal", "first@example.com"),
            ("personal", "second@example.com"),
        }
        assert {doc.metadata["scope"] for doc, _ in admin_combined} == {"global", "personal"}

        scopes_searched.clear()
        await retrieve_context(
            mock_query,
            mode="personal",
            user_email="admin@example.com",
            is_admin=True,
            personal_emails=owner_emails,
        )
        assert scopes_searched == [("personal", "admin@example.com")]

    print("Retrieval scope checks passed: personal=user-only, enterprise=shared, combined=both")

if __name__ == "__main__":
    asyncio.run(test_rag_retrieval_logic())
