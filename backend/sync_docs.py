
import logging
from pymysql.cursors import DictCursor
from database import get_db_connection, DB_NAME
from rag import is_qdrant_cloud_configured, vector_document_exists

logger = logging.getLogger(__name__)

def sync_document_status():
    """Verify that all 'ready' documents in the DB have corresponding vector data."""
    is_qdrant_cloud_configured()

    conn = get_db_connection(DB_NAME)
    cursor = conn.cursor(DictCursor)

    try:
        cursor.execute("SELECT id, filename, scope, owner_email FROM documents WHERE status = 'ready'")
        docs = cursor.fetchall()

        missing_documents = []
        for doc in docs:
            exists = vector_document_exists(doc["filename"], doc["scope"], doc["owner_email"])

            if not exists:
                logger.warning(f"Document '{doc['filename']}' (ID: {doc['id']}) is marked READY but vector data is missing. Marking as FAILED.")
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
