
import requests
import os

# Login to get token
login_url = "http://localhost:8007/login"
login_data = {"email": "anujsingh@gmail.com", "password": "password123"} # Assuming this is the password from previous logs
response = requests.post(login_url, json=login_data)
if response.status_code != 200:
    print(f"Login failed: {response.text}")
    exit()

token = response.json().get("access_token")
headers = {"Authorization": f"Bearer {token}"}

# Upload a test file
upload_url = "http://localhost:8007/upload?email=anujsingh@gmail.com&scope=personal"
file_content = "The secret code for the nexus engine is 998877. This is a highly confidential document."
with open("test.txt", "w") as f:
    f.write(file_content)

with open("test.txt", "rb") as f:
    files = {"file": ("test.txt", f, "text/plain")}
    response = requests.post(upload_url, headers=headers, files=files)

print(f"Upload status: {response.status_code}")
print(f"Upload response: {response.json()}")

if response.status_code == 200:
    # Ask a question
    ask_url = "http://localhost:8007/ask"
    ask_data = {"email": "anujsingh@gmail.com", "question": "What is the secret code for the nexus engine?", "mode": "personal"}
    response = requests.post(ask_url, headers=headers, json=ask_data)
    print(f"Ask status: {response.status_code}")
    print(f"Ask response: {response.json()}")

os.remove("test.txt")
