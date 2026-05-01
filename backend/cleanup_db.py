import database

def cleanup():
    conn = database.conn
    cursor = conn.cursor()
    
    # 1. Clear stuck processing
    cursor.execute("UPDATE documents SET status = 'failed', error_message = 'System reset. Please re-upload.' WHERE status = 'processing'")
    
    # 2. Clear failed documents if requested (optional, but good for a fresh start)
    # cursor.execute("DELETE FROM documents WHERE status = 'failed'")
    
    conn.commit()
    print("Cleanup successful. Processing documents have been marked as failed.")

if __name__ == "__main__":
    cleanup()
