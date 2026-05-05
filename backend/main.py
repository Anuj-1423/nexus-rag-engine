import os
import mysql.connector
import logging
import shutil
import asyncio
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Response, Header, Depends
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from database import get_db_connection, DB_NAME, init_db
from rag import ingest_document, generate_rag_response, delete_document_from_vector

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Enterprise Brain RAG API")

@app.get("/health")
def health_check():
    return {"status": "healthy"}

# Ensure directories exist
os.makedirs("temp_uploads", exist_ok=True)
os.makedirs("storage/vectors", exist_ok=True)
os.makedirs("storage/profiles", exist_ok=True)

app.mount("/profiles", StaticFiles(directory="storage/profiles"), name="profiles")

@app.on_event("startup")
def startup_event():
    try:
        logger.info("Starting up: Initializing database...")
        init_db()
        logger.info("Database initialized successfully.")
    except Exception as e:
        logger.error(f"FATAL: Database initialization failed: {e}")
        # We don't raise here so the server can at least start and show a health check,
        # but the app won't work until DB is fixed.

SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.txt', '.csv'}

class RegisterRequest(BaseModel):
    full_name: str
    email: str
    password: str
    role: str = "employee"

class LoginRequest(BaseModel):
    email: str
    password: str

from auth import hash_password, verify_password, create_token, verify_token

async def get_current_user(authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(401, "Authorization header missing")
    token = authorization.replace("Bearer ", "")
    payload = verify_token(token)
    if not payload:
        raise HTTPException(401, "Invalid or expired session. Please login again.")
    return payload

def verify_owner(email: str, current_user: dict):
    if email != current_user.get("email") and current_user.get("role") != "admin":
        raise HTTPException(403, "Access denied: You do not own this resource")

@app.post('/register')
def register(data: RegisterRequest):
    try:
        conn = get_db_connection(DB_NAME)
        cursor = conn.cursor()
        
        # Use hashed password
        hashed = hash_password(data.password)
        
        cursor.execute("INSERT INTO users (full_name, email, password_hash, role) VALUES (%s, %s, %s, %s)",
                    (data.full_name, data.email, hashed, data.role))
        conn.commit()
        return {"msg": "User registered successfully"}
    except mysql.connector.Error as db_err:
        logger.error(f"Registration DB Error: {db_err}")
        if "Duplicate entry" in str(db_err):
            raise HTTPException(400, "Email already registered")
        raise HTTPException(500, f"Database error during registration: {db_err}")
    except Exception as e:
        logger.error(f"Registration Error: {e}")
        raise HTTPException(500, f"Registration failed: {str(e)}")
    finally:
        if 'cursor' in locals(): cursor.close()
        if 'conn' in locals(): conn.close()

@app.post('/login')
def login(data: LoginRequest):
    try:
        conn = get_db_connection(DB_NAME)
        cursor = conn.cursor()
        
        # We must fetch by email first to verify hash
        cursor.execute("SELECT email, role, full_name, is_blocked, profile_pic, password_hash FROM users WHERE email = %s",
                    (data.email,))
        user = cursor.fetchone()
        
        if not user:
            raise HTTPException(401, "Invalid email or password")
            
        # Verify hash
        stored_hash = user[5]
        if not verify_password(data.password, stored_hash):
            # Fallback for plaintext (optional, for transition)
            if data.password == stored_hash:
                logger.warning(f"Plaintext login detected for {data.email}. Updating to hash.")
                # Optional: Update to hash here
            else:
                raise HTTPException(401, "Invalid email or password")
        
        token = create_token({"email": user[0], "role": user[1]})
        
        return {
            "token": token, 
            "email": user[0], 
            "role": user[1], 
            "name": user[2], 
            "is_blocked": bool(user[3]), 
            "profile_pic": user[4]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login Error: {e}")
        raise HTTPException(500, f"Login failed: {str(e)}")
    finally:
        if 'cursor' in locals(): cursor.close()
        if 'conn' in locals(): conn.close()

@app.get('/profile/{email}')
def get_profile(email: str, current_user: dict = Depends(get_current_user)):
    verify_owner(email, current_user)
    conn = get_db_connection(DB_NAME)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT full_name, email, phone, address, profile_pic, role FROM users WHERE email = %s", (email,))
        row = cursor.fetchone()
        if not row:
            raise HTTPException(404, "User not found")
        return {
            "full_name": row[0] or "",
            "email": row[1],
            "phone": row[2] or "",
            "address": row[3] or "",
            "profile_pic": row[4] or "",
            "role": row[5]
        }
    finally:
        cursor.close()
        conn.close()

@app.post('/profile')
async def update_profile(
    email: str = Form(...),
    full_name: str = Form(...),
    phone: str = Form(""),
    address: str = Form(""),
    profile_pic: Optional[UploadFile] = File(None),
    current_user: dict = Depends(get_current_user)
):
    verify_owner(email, current_user)
    conn = get_db_connection(DB_NAME)
    cursor = conn.cursor()
    try:
        if profile_pic and profile_pic.filename:
            pic_filename = f"{email.replace('@', '_').replace('.', '_')}_{profile_pic.filename}"
            pic_path = os.path.join("storage/profiles", pic_filename)
            with open(pic_path, "wb") as buffer:
                shutil.copyfileobj(profile_pic.file, buffer)
            
            cursor.execute(
                "UPDATE users SET full_name = %s, phone = %s, address = %s, profile_pic = %s WHERE email = %s",
                (full_name, phone, address, pic_filename, email)
            )
            conn.commit()
            return {"msg": "Profile updated successfully", "profile_pic": pic_filename}
        else:
            cursor.execute(
                "UPDATE users SET full_name = %s, phone = %s, address = %s WHERE email = %s",
                (full_name, phone, address, email)
            )
            conn.commit()
            return {"msg": "Profile updated successfully"}
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        cursor.close()
        conn.close()

@app.post('/upload')
async def upload_document(email: str, scope: str = "global", file: UploadFile = File(...), current_user: dict = Depends(get_current_user)):
    verify_owner(email, current_user)
    # 1. Auth & Initial Validation
    conn = get_db_connection(DB_NAME)
    cursor = conn.cursor()
    doc_id = None
    try:
        cursor.execute("SELECT role, is_blocked FROM users WHERE email = %s", (email,))
        user_data = cursor.fetchone()
        if user_data and user_data[1]:
            raise HTTPException(403, "Your account is suspended. You cannot upload documents.")
        if scope == 'global' and (not user_data or user_data[0] != 'admin'):
            raise HTTPException(403, "Only admins can upload global documents")

        filename = file.filename or "untitled"
        ext = os.path.splitext(filename)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise HTTPException(400, "Unsupported file type")

        file_bytes = await file.read()

        cursor.execute(
            "INSERT INTO documents (filename, file_type, file_size_bytes, status, owner_email, scope) VALUES (%s, %s, %s, %s, %s, %s)",
            (filename, ext.lstrip("."), len(file_bytes), "processing", email, scope)
        )
        conn.commit()
        doc_id = cursor.lastrowid
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        cursor.close()
        conn.close()

    # 2. AI Processing
    try:
        result = ingest_document(file_bytes, filename, scope=scope, user_email=email)

        conn2 = get_db_connection(DB_NAME)
        cursor2 = conn2.cursor()
        try:
            cursor2.execute(
                """UPDATE documents SET doc_title = %s, total_chunks = %s, total_sections = %s, total_pages = %s, status = 'ready' WHERE id = %s""",
                (result.get("doc_title"), result.get("total_chunks"), result.get("total_sections"), result.get("total_pages"), doc_id)
            )
            conn2.commit()
        finally:
            cursor2.close()
            conn2.close()

        return {"msg": "success", "status": "ready"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ingest failure: {e}")
        conn3 = get_db_connection(DB_NAME)
        cursor3 = conn3.cursor()
        try:
            cursor3.execute("UPDATE documents SET status = 'failed', error_message = %s WHERE id = %s", (str(e)[:500], doc_id))
            conn3.commit()
        finally:
            cursor3.close()
            conn3.close()
        raise HTTPException(500, f"AI Processing Failed: {str(e)}")

class AskRequest(BaseModel):
    email: str
    question: str
    mode: str = "combined"

@app.post('/ask')
async def ask_question(data: AskRequest, current_user: dict = Depends(get_current_user)):
    verify_owner(data.email, current_user)
    # Use to_thread for blocking database calls
    # Use to_thread for blocking database calls
    conn = await asyncio.to_thread(get_db_connection, DB_NAME)
    cursor = conn.cursor()
    try:
        await asyncio.to_thread(cursor.execute, "SELECT is_blocked FROM users WHERE email = %s", (data.email,))
        user_data = cursor.fetchone()
        if user_data and user_data[0]:
            raise HTTPException(403, "Your account is suspended. You cannot query the AI.")
            
        # Fetch recent chat history for context
        await asyncio.to_thread(
            cursor.execute,
            "SELECT question, answer FROM chats WHERE email = %s ORDER BY created_at ASC LIMIT 5",
            (data.email,)
        )
        history_rows = cursor.fetchall()
        chat_history = [{"question": r[0], "answer": r[1]} for r in history_rows]

        res = await generate_rag_response(data.question, mode=data.mode, user_email=data.email, chat_history=chat_history)
        
        await asyncio.to_thread(
            cursor.execute,
            "INSERT INTO chats (email, question, answer, scope) VALUES (%s, %s, %s, %s)",
            (data.email, data.question, res['answer'], data.mode)
        )
        await asyncio.to_thread(conn.commit)
        return res
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in ask_question: {e}")
        raise HTTPException(500, str(e))
    finally:
        await asyncio.to_thread(cursor.close)
        await asyncio.to_thread(conn.close)

@app.get('/admin/stats')
def admin_stats(current_user: dict = Depends(get_current_user)):
    if current_user.get("role") != "admin":
        raise HTTPException(403, "Access denied: Admin role required")
        
    conn = get_db_connection(DB_NAME)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT COUNT(*) FROM users")
        u = cursor.fetchone()[0]
        cursor.execute("SELECT scope, COUNT(*) FROM documents WHERE status = 'ready' GROUP BY scope")
        d = dict(cursor.fetchall())
        cursor.execute("SELECT COUNT(*) FROM chats")
        q = cursor.fetchone()[0]
        cursor.execute("SELECT email, COUNT(*) as count FROM chats GROUP BY email ORDER BY count DESC LIMIT 5")
        top = [{"email": r[0], "queries": r[1]} for r in cursor.fetchall()]
        return {"total_users": u, "global_docs": d.get("global", 0), "personal_docs": d.get("personal", 0), "total_queries": q, "top_users": top}
    finally:
        cursor.close()
        conn.close()

class BlockRequest(BaseModel):
    is_blocked: bool

@app.get('/admin/users')
def get_all_users(current_user: dict = Depends(get_current_user)):
    if current_user.get("role") != "admin":
        raise HTTPException(403, "Access denied: Admin role required")
        
    conn = get_db_connection(DB_NAME)
    cursor = conn.cursor()
    try:
        # Role check already done
        cursor.execute("SELECT email, full_name, role, is_blocked FROM users ORDER BY created_at DESC")
        return [{"email": r[0], "name": r[1], "role": r[2], "is_blocked": bool(r[3])} for r in cursor.fetchall()]
    finally:
        cursor.close()
        conn.close()

@app.post('/admin/users/{user_email}/block')
def toggle_block_user(user_email: str, data: BlockRequest, current_user: dict = Depends(get_current_user)):
    if current_user.get("role") != "admin":
        raise HTTPException(403, "Access denied: Admin role required")
        
    conn = get_db_connection(DB_NAME)
    cursor = conn.cursor()
    try:
        # Role check already done
        
        cursor.execute("UPDATE users SET is_blocked = %s WHERE email = %s", (data.is_blocked, user_email))
        conn.commit()
        return {"msg": "User block status updated"}
    finally:
        cursor.close()
        conn.close()

@app.get('/admin/users/{user_email}/details')
def get_user_details(user_email: str, current_user: dict = Depends(get_current_user)):
    if current_user.get("role") != "admin":
        raise HTTPException(403, "Access denied: Admin role required")
        
    conn = get_db_connection(DB_NAME)
    cursor = conn.cursor()
    try:
        # Role check already done
            
        # Get user details
        cursor.execute("SELECT full_name, email, phone, address, profile_pic, role, is_blocked FROM users WHERE email = %s", (user_email,))
        row = cursor.fetchone()
        if not row:
            raise HTTPException(404, "User not found")
            
        # Get doc count
        cursor.execute("SELECT COUNT(*) FROM documents WHERE owner_email = %s", (user_email,))
        doc_count = cursor.fetchone()[0]
        
        # Get query count
        cursor.execute("SELECT COUNT(*) FROM chats WHERE email = %s", (user_email,))
        query_count = cursor.fetchone()[0]
        
        return {
            "full_name": row[0] or "",
            "email": row[1],
            "phone": row[2] or "",
            "address": row[3] or "",
            "profile_pic": row[4] or "",
            "role": row[5],
            "is_blocked": bool(row[6]),
            "doc_count": doc_count,
            "query_count": query_count
        }
    finally:
        cursor.close()
        conn.close()

@app.get('/documents/{email}')
def list_docs(email: str, response: Response, current_user: dict = Depends(get_current_user)):
    verify_owner(email, current_user)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    conn = get_db_connection(DB_NAME)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT filename, scope, status, uploaded_at FROM documents WHERE scope = 'global' OR owner_email = %s ORDER BY uploaded_at DESC", (email,))
        return [{"filename": r[0], "scope": r[1], "status": r[2], "date": str(r[3])} for r in cursor.fetchall()]
    finally:
        cursor.close()
        conn.close()

@app.get('/history/{email}')
def get_history(email: str, current_user: dict = Depends(get_current_user)):
    verify_owner(email, current_user)
    conn = get_db_connection(DB_NAME)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT question, answer, scope, created_at FROM chats WHERE email = %s ORDER BY created_at DESC", (email,))
        return [{"question": r[0], "answer": r[1], "scope": r[2], "date": str(r[3])} for r in cursor.fetchall()]
    finally:
        cursor.close()
        conn.close()

@app.delete('/documents/{email}/{filename}')
def delete_doc(email: str, filename: str, scope: str = "global", current_user: dict = Depends(get_current_user)):
    verify_owner(email, current_user)
    conn = get_db_connection(DB_NAME)
    cursor = conn.cursor()
    try:
        # 1. Delete from MySQL Database
        if scope == "global":
            cursor.execute("DELETE FROM documents WHERE filename = %s AND scope = 'global'", (filename,))
        else:
            cursor.execute("DELETE FROM documents WHERE filename = %s AND owner_email = %s AND scope = 'personal'", (filename, email))
        conn.commit()
        
        # 2. Delete from ChromaDB Vector Store
        vector_success = delete_document_from_vector(filename, scope, email)
        
        if vector_success:
            return {"status": "success", "message": f"Deleted {filename} from both Database and Vector Store."}
        else:
            return {"status": "warning", "message": f"Deleted from Database, but Vector deletion returned False (might not exist)."}
    except Exception as e:
        logger.error(f"Deletion error: {e}")
        raise HTTPException(500, str(e))
    finally:
        cursor.close()
        conn.close()

@app.get("/{file_path:path}")
async def serve_frontend(file_path: str):
    if file_path == "": file_path = "index.html"
    full_path = os.path.join("../frontend", file_path)
    if os.path.isfile(full_path): return FileResponse(full_path)
    if not file_path.endswith(".html"):
        hp = full_path + ".html"
        if os.path.isfile(hp): return FileResponse(hp)
    raise HTTPException(404, "Not Found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)