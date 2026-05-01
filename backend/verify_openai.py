import os
from openai import OpenAI
from dotenv import load_dotenv

def verify_openai():
    # Force reload from .env and override any existing environment variables
    load_dotenv(override=True)
    
    api_key = os.getenv("OPENAI_API_KEY")
    
    print("\n--- OpenAI Diagnostic (USER SPECIFIC FORMAT) ---")
    
    if not api_key:
        print("[ERROR] OPENAI_API_KEY not found.")
        return False

    print(f"[INFO] Using key starting with: {api_key[:12]}...")

    try:
        client = OpenAI(api_key=api_key)
        print("[INFO] Testing with gpt-5.4-mini and responses.create...")
        
        # EXACT CODE FORMAT REQUESTED BY USER
        response = client.responses.create(
            model="gpt-5.4-mini",
            input="write a haiku about ai",
            store=True,
        )
        print(f"[RESULT] {response.output_text}")
        print("[SUCCESS] Your specific OpenAI format is WORKING!")
        return True
    except Exception as e:
        print(f"[FAILED] Error with requested format: {e}")
        return False

if __name__ == "__main__":
    verify_openai()
