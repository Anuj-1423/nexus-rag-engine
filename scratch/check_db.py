
import os
import mysql.connector

DB_HOST = os.getenv('DB_HOST', 'localhost').strip()
DB_USER = os.getenv('DB_USER', 'root').strip()
DB_PASSWORD = os.getenv('DB_PASSWORD', '').strip()
DB_NAME = os.getenv('DB_NAME', 'defaultdb').strip()
DB_PORT_RAW = os.getenv('DB_PORT', '3306').strip()

if '://' in DB_HOST:
    DB_HOST = DB_HOST.split('://')[-1].split('@')[-1].split(':')[0].split('/')[0]

try:
    DB_PORT = int(DB_PORT_RAW)
except ValueError:
    DB_PORT = 3306

config = {
    'host': DB_HOST,
    'user': DB_USER,
    'password': DB_PASSWORD,
    'database': DB_NAME,
    'port': DB_PORT,
    'autocommit': True,
    'connect_timeout': 10,
    'use_pure': True,
}

if DB_HOST not in ('localhost', '127.0.0.1'):
    config['ssl_disabled'] = False
    config['ssl_verify_cert'] = False

print(f"Connecting to MySQL at {DB_HOST}:{DB_PORT}/{DB_NAME} as {DB_USER}...")

try:
    conn = mysql.connector.connect(**config)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM documents')
    rows = cursor.fetchall()
    print(f"Found {len(rows)} documents in the database.")
    for row in rows:
        print(row)
    cursor.close()
    conn.close()
except Exception as e:
    print(f"Error connecting to MySQL at {DB_HOST}:{DB_PORT}/{DB_NAME}: {e}")
    print('Check that DB_HOST, DB_USER, DB_PASSWORD, DB_NAME, and DB_PORT are set correctly for your deployment.')
