import jwt, datetime
import bcrypt
SECRET='secret123'

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
    payload['exp'] = datetime.datetime.utcnow()+datetime.timedelta(hours=12)
    return jwt.encode(payload, SECRET, algorithm='HS256')