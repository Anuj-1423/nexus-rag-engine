
import requests
import os

# Register a new user
register_url = "http://localhost:8007/register"
reg_data = {"full_name": "Test User", "email": "test@example.com", "password": "password123", "role": "admin"}
requests.post(register_url, json=reg_data)

# Login
login_url = "http://localhost:8007/login"
login_data = {"email": "test@example.com", "password": "password123"}
response = requests.post(login_url, json=login_data)
if response.status_code != 200:
    print(f"Login failed: {response.text}")
    exit()

token = response.json().get("access_token")
headers = {"Authorization": f"Bearer {token}"}

# Upload
upload_url = "http://localhost:8007/upload?email=test@example.com&scope=personal"
file_content = "The secret code for the nexus engine is 998877. This is a highly confidential document."
with open("test.txt", "w") as f:
    f.write(file_content)

with open("test.txt", "rb") as f:
    files = {"file": ("test.txt", f, "text/plain")}
    response = requests.post(upload_url, headers=headers, files=files)

print(f"Upload response: {response.json()}")

if response.status_code == 200:
    # Ask
    ask_url = "http://localhost:8007/ask"
    ask_data = {"email": "test@example.com", "question": "What is the secret code for the nexus engine?", "mode": "personal"}
    response = requests.post(ask_url, headers=headers, json=ask_data)
    print(f"Ask response: {response.json()}")

os.remove("test.txt")
