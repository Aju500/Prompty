import os
import requests
from dotenv import load_dotenv

load_dotenv()

# We check for the token first to provide a clear error message if it's missing.
HF_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
if not HF_TOKEN:
    raise ValueError("HUGGINGFACE_API_TOKEN not found in .env file. Please check your environment variables.")

API_URL = "https://router.huggingface.co/v1/chat/completions"
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

def query(prompt: str) -> str:
    """
    Sends a prompt to the new Hugging Face Chat Completions API using Mistral.
    """
    payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": prompt}]
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        
        output = response.json()
        if 'choices' in output and output['choices'] and 'message' in output['choices'][0] and 'content' in output['choices'][0]['message']:
            return output['choices'][0]['message']['content'].strip()
        else:
            raise ValueError(f"Unexpected API response format: {output}")

    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"[Mistral Network Error] {e}")
    except Exception as e:
        raise RuntimeError(f"[Mistral Error] {e}")
