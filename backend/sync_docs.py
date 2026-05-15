
import os
import logging
import hashlib
from database import get_db_connection, DB_NAME
from rag import get_index_path

logger = logging.getLogger(__name__)

def sync_document_status():
    """Verify that all 'ready' documents in the DB have corresponding vector data."""
    conn = get_db_connection(DB_NAME)
    cursor = conn.cursor(dictionary=True)
    
    try:
        cursor.execute("SELECT id, filename, scope, owner_email FROM documents WHERE status = 'ready'")
        docs = cursor.fetchall()
        
        broken_count = 0
        for doc in docs:
            index_path = get_index_path(doc['scope'], doc['owner_email'])
            
            # Check if directory exists and contains chroma data
            exists = False
            if os.path.exists(index_path):
                # Chroma usually has a sqlite3 file
                if any(f.endswith('.sqlite3') for f in os.listdir(index_path)):
                    exists = True
            
            if not exists:
                logger.warning(f"Document '{doc['filename']}' (ID: {doc['id']}) is marked READY but vector path {index_path} is missing or empty. Marking as FAILED.")
                cursor.execute(
                    "UPDATE documents SET status = 'failed', error_message = 'Vector data missing. Please re-upload.' WHERE id = %s",
                    (doc['id'],)
                )
                broken_count += 1
        
        if broken_count > 0:
            conn.commit()
            logger.info(f"Sync complete. Fixed {broken_count} stale document records.")
        else:
            logger.info("Sync complete. All document records are consistent with vector storage.")
            
    except Exception as e:
        logger.error(f"Error during document status sync: {e}")
    finally:
        cursor.close()
        conn.close()
