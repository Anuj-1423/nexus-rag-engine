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
    max_retries = 5
    retry_delay = 3
    
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
                'connection_timeout': 10
            }
            
            # SSL Configuration for Cloud (Aiven/Render)
            # Aiven requires SSL. We use ssl_mode='REQUIRED' which is standard.
            if DB_HOST != 'localhost' and DB_HOST != '127.0.0.1':
                config['ssl_mode'] = 'REQUIRED'
                # If we have a CA file, we could use it, but REQUIRED usually works with system certs
                # config['ssl_ca'] = os.getenv('DB_SSL_CA') 
            
            if database:
                config['database'] = database
                
            return mysql.connector.connect(**config)
        except mysql.connector.Error as err:
            last_err = err
            logger.warning(f"Database connection attempt {i+1} failed (Host: {DB_HOST}, Port: {DB_PORT}). Retrying in {retry_delay}s... Error: {err}")
            time.sleep(retry_delay)
    
    logger.error(f"CRITICAL: Database connection failed after {max_retries} attempts: {last_err}")
    raise Exception(f"Could not connect to MySQL at {DB_HOST}:{DB_PORT}. Please check your environment variables and Aiven status. Error: {last_err}")

def init_db():
    """Initializes the database and tables if they don't exist."""
    try:
        logger.info(f"Connecting to MySQL at {DB_HOST}:{DB_PORT} to ensure database '{DB_NAME}' exists...")
        
        # 1. Create Database
        try:
            conn_base = get_db_connection()
            cursor = conn_base.cursor()
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{DB_NAME}` DEFAULT CHARACTER SET 'utf8mb4'")
            cursor.close()
            conn_base.close()
        except Exception as e:
            logger.warning(f"Note: Could not run 'CREATE DATABASE' (might be lack of permissions). Continuing... Error: {e}")

        # 2. Create Tables
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

if __name__ == "__main__":
    init_db()