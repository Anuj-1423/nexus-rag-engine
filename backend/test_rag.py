import os
import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from rag import ingest_document, get_embeddings
    import google.generativeai as genai
except ImportError as e:
    logger.error(f"Import error: {e}")
    sys.exit(1)

# Test API Key
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    logger.error("GOOGLE_API_KEY not found in environment.")
    sys.exit(1)

# Test Embeddings
try:
    logger.info("Testing Embeddings...")
    emb = get_embeddings()
    test_text = ["Hello world"]
    vectors = emb.embed_documents(test_text)
    logger.info(f"Successfully embedded. Vector size: {len(vectors[0])}")
except Exception as e:
    logger.error(f"Embeddings failed: {e}")

# Test Ingestion with dummy data
try:
    logger.info("Testing Ingestion...")
    dummy_pdf = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/Resources << >>\n/Contents 4 0 R\n>>\nendobj\n4 0 obj\n<< /Length 10 >>\nstream\nHello World\nendstream\nendobj\ntrailer\n<<\n/Root 1 0 R\n>>\n%%EOF"
    ingest_document(dummy_pdf, "test.pdf", scope="global")
    logger.info("Ingestion successful.")
except Exception as e:
    logger.error(f"Ingestion failed: {e}")
