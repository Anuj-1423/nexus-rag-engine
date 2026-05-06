import os
import mysql.connector
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database Configuration
DB_HOST = os.getenv('DB_HOST', 'localhost').strip()
DB_USER = os.getenv('DB_USER', 'root').strip()
DB_PASSWORD = os.getenv('DB_PASSWORD', 'Ishan@1423').strip()
DB_NAME = os.getenv('DB_NAME', 'defaultdb').strip()
DB_PORT_RAW = os.getenv('DB_PORT', '3306').strip()

# Robustness: Handle accidental 'mysql://' prefix in DB_HOST
if "://" in DB_HOST:
    DB_HOST = DB_HOST.split("://")[-1].split("@")[-1].split(":")[0].split("/")[0]

try:
    DB_PORT = int(DB_PORT_RAW)
except:
    DB_PORT = 3306

def get_db_connection(database=None):
    """Creates a fresh connection to the MySQL server with retries and SSL support."""
    import time
    max_retries = 3
    retry_delay = 2
    
    last_err = None
    for i in range(max_retries):
        try:
            # Base config
            config = {
                'host': DB_HOST,
                'user': DB_USER,
                'password': DB_PASSWORD,
                'port': DB_PORT,
                'autocommit': True,
                'connect_timeout': 10,
                'use_pure': True  # Force Pure Python for best compatibility on Render
            }
            
            if database:
                config['database'] = database
            
            # SSL Configuration for Cloud (Aiven/Render)
            if DB_HOST not in ['localhost', '127.0.0.1']:
                # ssl_mode is often unsupported in newer C-extensions, 
                # using ssl_disabled and ssl_verify_cert is more reliable.
                config['ssl_disabled'] = False
                config['ssl_verify_cert'] = False

            return mysql.connector.connect(**config)

        except mysql.connector.Error as err:
            last_err = err
            logger.warning(f"Database connection attempt {i+1} failed. Error: {err}")
            time.sleep(retry_delay)
    
    logger.error(f"CRITICAL: Database connection failed: {last_err}")
    raise Exception(f"Database connection failed: {last_err}")

def init_db():
    """Initializes the database and tables if they don't exist."""
    try:
        logger.info(f"Connecting to MySQL at {DB_HOST}:{DB_PORT} to ensure database '{DB_NAME}' exists...")
        
        # 1. Create Database
        try:
            # Connect to server WITHOUT database to create it
            conn_base = get_db_connection()
            cursor = conn_base.cursor()
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{DB_NAME}` DEFAULT CHARACTER SET 'utf8mb4'")
            cursor.close()
            conn_base.close()
            logger.info(f"Database '{DB_NAME}' verified/created.")
        except Exception as e:
            logger.warning(f"Note: Could not run 'CREATE DATABASE' (might be lack of permissions). Continuing... Error: {e}")

        # 2. Create Tables
        # Now connect WITH the database
        conn = get_db_connection(DB_NAME)
        cursor = conn.cursor()
        
        # User Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                full_name VARCHAR(255),
                email VARCHAR(255) UNIQUE,
                password_hash VARCHAR(255),
                role VARCHAR(50),
                is_blocked BOOLEAN DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Documents Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INT AUTO_INCREMENT PRIMARY KEY,
                filename VARCHAR(255),
                file_type VARCHAR(50),
                file_size_bytes BIGINT,
                doc_title VARCHAR(255),
                total_chunks INT DEFAULT 0,
                total_sections INT DEFAULT 0,
                total_pages INT DEFAULT 0,
                status VARCHAR(50),
                error_message TEXT,
                owner_email VARCHAR(255),
                scope VARCHAR(50),
                uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Chat History Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chats (
                id INT AUTO_INCREMENT PRIMARY KEY,
                email VARCHAR(255),
                question TEXT,
                answer TEXT,
                scope VARCHAR(50),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()

        # 3. Migrations Helper
        def ensure_column(table, column, definition):
            mig_conn = get_db_connection(DB_NAME)
            check_cursor = mig_conn.cursor()
            try:
                check_cursor.execute(f"SELECT {column} FROM {table} LIMIT 1")
                check_cursor.fetchall()
            except mysql.connector.Error:
                logger.info(f"Migration: Adding {column} to {table}")
                alter_cursor = mig_conn.cursor()
                alter_cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
                mig_conn.commit()
                alter_cursor.close()
            finally:
                check_cursor.close()
                mig_conn.close()

        # Run Migrations
        ensure_column("users", "is_blocked", "BOOLEAN DEFAULT 0")
        ensure_column("users", "phone", "VARCHAR(50)")
        ensure_column("users", "address", "TEXT")
        ensure_column("users", "profile_pic", "VARCHAR(255)")
        ensure_column("documents", "total_sections", "INT DEFAULT 0")
        ensure_column("documents", "total_chunks", "INT DEFAULT 0")
        ensure_column("documents", "scope", "VARCHAR(50) DEFAULT 'global'")
        ensure_column("documents", "owner_email", "VARCHAR(255)")
        ensure_column("chats", "scope", "VARCHAR(50) DEFAULT 'global'")

        cursor.close()
        conn.close()
        logger.info("Database system initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise e

if __name__ == "__main__":
    init_db()