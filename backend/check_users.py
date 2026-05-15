
from database import get_db_connection, DB_NAME

def check_users():
    conn = get_db_connection(DB_NAME)
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT email, role, is_blocked FROM users")
        users = cursor.fetchall()
        print(f"{'Email':<30} | {'Role':<10} | {'Blocked'}")
        print("-" * 50)
        for user in users:
            print(f"{user['email']:<30} | {user['role']:<10} | {user['is_blocked']}")
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    check_users()
