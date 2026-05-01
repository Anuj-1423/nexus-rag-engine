import bcrypt
from database import conn
from auth import verify_password

def debug():
    c = conn.cursor()
    c.execute("SELECT email, password, role FROM users WHERE email = 'anuj@gmail.com'")
    user = c.fetchone()
    if not user:
        print("User not found!")
        return
    
    email, hashed, role = user
    print(f"Email: {email}")
    print(f"Role: {role}")
    print(f"Hashed password in DB: {hashed}")
    
    test_pass = "admin123"
    is_valid = verify_password(test_pass, hashed)
    print(f"Testing password '{test_pass}': {'VALID' if is_valid else 'INVALID'}")
    
    # Check if we can hash and verify in one go
    new_hash = bcrypt.hashpw(test_pass.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    is_valid_new = verify_password(test_pass, new_hash)
    print(f"Testing fresh hash: {'VALID' if is_valid_new else 'INVALID'}")

if __name__ == "__main__":
    debug()
