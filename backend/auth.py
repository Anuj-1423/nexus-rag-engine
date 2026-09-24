import os
import jwt
import datetime
import bcrypt

SECRET = (
    os.getenv("JWT_SECRET")
    or os.getenv("APP_SECRET_KEY")
    or "change-me-to-a-long-random-secret-key-32-bytes-plus"
)

# Ensure a secure length for HS256 validation and avoid Render warnings.
if len(SECRET.encode("utf-8")) < 32:
    SECRET = SECRET + "-" * max(0, 32 - len(SECRET.encode("utf-8")))


def hash_password(p):
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(p.encode('utf-8'), salt).decode('utf-8')


def verify_password(p, h):
    try:
        return bcrypt.checkpw(p.encode('utf-8'), h.encode('utf-8'))
    except Exception:
        return False


def create_token(data):
    payload = data.copy()
    payload['exp'] = datetime.datetime.utcnow() + datetime.timedelta(hours=12)
    return jwt.encode(payload, SECRET, algorithm='HS256')


def verify_token(token):
    try:
        return jwt.decode(token, SECRET, algorithms=['HS256'])
    except Exception:
        return None