from database import get_db_connection, DB_NAME
import mysql.connector

def sync_schema():
    conn = get_db_connection(DB_NAME)
    cursor = conn.cursor()
    
    print("Syncing Users table...")
    
    # 1. Rename 'password' to 'password_hash' if it exists
    try:
        cursor.execute("ALTER TABLE users CHANGE COLUMN password password_hash VARCHAR(255)")
        print("Renamed 'password' to 'password_hash'")
    except:
        pass

    # 2. Add 'full_name' if missing (already done but safe)
    try:
        cursor.execute("ALTER TABLE users ADD COLUMN full_name VARCHAR(255) AFTER id")
        print("Added 'full_name'")
    except:
        pass

    # 3. Drop redundant 'name' if it exists
    try:
        cursor.execute("ALTER TABLE users DROP COLUMN name")
        print("Dropped redundant 'name' column")
    except:
        pass

    # 4. Add 'created_at' if missing
    try:
        cursor.execute("ALTER TABLE users ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
        print("Added 'created_at'")
    except:
        pass

    conn.commit()
    cursor.close()
    conn.close()
    print("Sync Complete!")

if __name__ == "__main__":
    sync_schema()
