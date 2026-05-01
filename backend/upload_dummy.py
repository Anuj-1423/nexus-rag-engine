import requests
import os

BASE_URL = "http://localhost:8000"

def upload_file(email, scope, filename, content):
    print(f"Uploading {filename} as {email} (Scope: {scope})...")
    with open(filename, "w") as f:
        f.write(content)
    
    # Use params for query parameters
    params = {'email': email, 'scope': scope}
    
    try:
        # Re-open file in binary mode for the request
        with open(filename, 'rb') as f_bin:
            files = {'file': (filename, f_bin)}
            response = requests.post(f"{BASE_URL}/upload", params=params, files=files)
            print(f"Response: {response.status_code} - {response.text}")
    finally:
        if os.path.exists(filename):
            os.remove(filename)

if __name__ == "__main__":
    # 1. Admin Upload (Enterprise)
    upload_file(
        email="anuj@gmail.com", 
        scope="global", 
        filename="enterprise_policy.txt", 
        content="The official company policy is that every employee gets 25 days of annual leave and free coffee."
    )
    
    # 2. User Upload (Personal)
    upload_file(
        email="anujsingh@gmail.com", 
        scope="personal", 
        filename="personal_notes.txt", 
        content="My personal secret code for the vault is ALFA-99 and my favorite food is Pizza."
    )
