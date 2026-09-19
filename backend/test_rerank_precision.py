from langchain_core.documents import Document
from reranker import rerank


def test_fallback_rerank_prefers_relevant_document():
    docs = [
        Document(page_content="Finance budget summary for Q4 operations was approved.", metadata={"filename": "budget.txt"}),
        Document(page_content="Enterprise password policy requires MFA and 12-character minimum passwords.", metadata={"filename": "security.txt"}),
    ]

    ranked = rerank("What is the enterprise password policy?", docs, top_n=2)
    assert ranked[0][0].metadata["filename"] == "security.txt"


if __name__ == "__main__":
    test_fallback_rerank_prefers_relevant_document()
    print("fallback rerank precision test passed")
