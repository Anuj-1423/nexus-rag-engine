import os
from google import genai
from dotenv import load_dotenv

# Try to load from .env if it exists
load_dotenv()

def verify_api_key():
    api_key = os.getenv("GOOGLE_API_KEY")
    print(f"\n--- API Key Diagnostic ---")
    
    if not api_key:
        print("❌ ERROR: GOOGLE_API_KEY not found in environment or .env file.")
        print("👉 Action: Run 'set GOOGLE_API_KEY=your_key' in CMD or create a .env file.")
        return False
    
    print(f"✅ Key Found: {api_key[:5]}...{api_key[-4:]}")
    
    try:
        print("📡 Testing connection to Gemini AI...")
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents="Say 'API Key is Working!'"
        )
        print(f"✨ RESULT: {response.text.strip()}")
        print("✅ SUCCESS: Your API Key is valid and working!")
        return True
    except Exception as e:
        print(f"❌ CONNECTION FAILED: {e}")
        print("👉 Possible causes: Invalid key, restricted region, or network proxy.")
        return False

if __name__ == "__main__":
    verify_api_key()
