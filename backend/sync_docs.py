
import os
import logging
import hashlib
from database import get_db_connection, DB_NAME
from rag import get_index_path, is_chroma_cloud_configured, vector_document_exists

logger = logging.getLogger(__name__)

def sync_document_status():
    """Verify that all 'ready' documents in the DB have corresponding vector data."""
    cloud_storage = is_chroma_cloud_configured()
    if os.getenv("VERCEL") == "1" and not cloud_storage:
        raise RuntimeError("CHROMA_API_KEY, CHROMA_TENANT, and CHROMA_DATABASE are required on Vercel.")

    conn = get_db_connection(DB_NAME)
    cursor = conn.cursor(dictionary=True)

    try:
        cursor.execute("SELECT id, filename, scope, owner_email FROM documents WHERE status = 'ready'")
        docs = cursor.fetchall()

        missing_documents = []
        for doc in docs:
            if cloud_storage:
                exists = vector_document_exists(doc["filename"], doc["scope"], doc["owner_email"])
                index_path = "Chroma Cloud"
            else:
                index_path = get_index_path(doc['scope'], doc['owner_email'])
                exists = os.path.exists(index_path) and any(
                    filename.endswith('.sqlite3') for filename in os.listdir(index_path)
                ) if os.path.exists(index_path) else False

            if not exists:
                logger.warning(f"Document '{doc['filename']}' (ID: {doc['id']}) is marked READY but vector path {index_path} is missing or empty. Marking as FAILED.")
                missing_documents.append(doc["id"])

        if missing_documents:
            for document_id in missing_documents:
                cursor.execute(
                    "UPDATE documents SET status = 'failed', error_message = 'Vector data missing. Please re-upload.' WHERE id = %s",
                    (document_id,),
                )
            conn.commit()
            logger.info(f"Sync complete. Fixed {len(missing_documents)} stale document records.")
        else:
            logger.info("Sync complete. All document records are consistent with vector storage.")

    except Exception as e:
        logger.error(f"Error during document status sync: {e}")
    finally:
        cursor.close()
        conn.close()
