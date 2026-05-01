import os
import mysql.connector

print('DB_HOST=', os.getenv('DB_HOST'))
print('DB_USER=', os.getenv('DB_USER'))
print('DB_PASSWORD=', '*' * len(os.getenv('DB_PASSWORD') or ''))
print('DB_NAME=', os.getenv('DB_NAME', 'enterprise_rag'))

try:
    conn = mysql.connector.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        user=os.getenv('DB_USER', 'root'),
        password=os.getenv('DB_PASSWORD', ''),
        database=os.getenv('DB_NAME', 'enterprise_rag')
    )
    print('Connection successful')
    conn.close()
except Exception as e:
    print(type(e).__name__, e)
