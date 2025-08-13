import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

# The API key must be configured once for the entire library.
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def query(prompt: str) -> str:
    """
    Sends a prompt to the Google Gemini API using the correct methods.
    """
    try:
        # 1. Instantiate the specific model requested.
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # 2. Generate the content from the model instance.
        response = model.generate_content(prompt)
        
        # 3. Return the text from the response.
        return response.text.strip()
        
    except Exception as e:
        # Provides a helpful error if the API key is bad or the model name is inaccessible.
        raise RuntimeError(f"[Gemini Error] Failed to get response: {e}")
