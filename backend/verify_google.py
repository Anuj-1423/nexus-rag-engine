import os
from google import genai
from dotenv import load_dotenv

def verify_google():
    # Force clear from environment to ensure it reads from .env
    if "GOOGLE_API_KEY" in os.environ:
        del os.environ["GOOGLE_API_KEY"]
    
    load_dotenv(override=True)
    api_key = os.getenv("GOOGLE_API_KEY")
    
    print("\n--- Google API Diagnostic ---")
    
    if not api_key:
        print("[ERROR] GOOGLE_API_KEY not found.")
        return False
    
    print(f"[INFO] System is using key starting with: {api_key[:10]}...")
    
    try:
        print("[INFO] Testing Gemini connection...")
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents="Say 'Google is ready!'"
        )
        print(f"[RESULT] {response.text.strip()}")
        print("[SUCCESS] Google Gemini is fully configured!")
        return True
    except Exception as e:
        print(f"[FAILED] Connection error: {e}")
        return False

if __name__ == "__main__":
    verify_google()
