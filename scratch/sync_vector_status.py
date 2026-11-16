import os
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from dotenv import load_dotenv

load_dotenv(BACKEND_DIR / ".env")

from database import DB_NAME, get_db_connection
from rag import get_index_path


def sync_status():
    try:
        conn = get_db_connection(DB_NAME)
        cursor = conn.cursor()

        cursor.execute("SELECT id, filename, owner_email, scope, status FROM documents WHERE status = 'ready'")
        docs = cursor.fetchall()

        print(f"Checking {len(docs)} 'ready' documents...")

        for doc_id, filename, email, scope, status in docs:
            path = get_index_path(scope, email)
            exists = (
                os.path.exists(path)
                and any(entry.endswith(".sqlite3") for entry in os.listdir(path))
                if os.path.exists(path)
                else False
            )

            if not exists:
                print(f"FIX: Document {doc_id} ('{filename}') is marked 'ready' but vector data is missing. Marking it as failed.")
                cursor.execute(
                    "UPDATE documents SET status = 'failed', error_message = 'Vector data missing. Please re-upload.' WHERE id = %s",
                    (doc_id,),
                )
            else:
                print(f"OK: Document {doc_id} ('{filename}') has vector data.")

        conn.commit()
        cursor.close()
        conn.close()
        print("Sync complete.")
    except Exception as error:
        print(f"Error: {error}")


if __name__ == "__main__":
    sync_status()
