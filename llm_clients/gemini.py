from google import genai
from dotenv import load_dotenv

load_dotenv()
client = genai.Client()

def query(prompt: str) -> str:
    try:
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        return response.text.strip()
    except Exception as e:
        raise RuntimeError(f"[Gemini error] {e}")
