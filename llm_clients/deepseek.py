import os
import requests
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
if not HF_TOKEN:
    raise ValueError("HUGGINGFACE_API_TOKEN not found in .env file.")

API_URL = "https://router.huggingface.co/v1/chat/completions"
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B:cerebras"
headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}

def query(prompt: str) -> str:
    payload = {"model": MODEL_ID, "messages": [{"role": "user", "content": prompt}]}
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        output = response.json()
        if 'choices' in output and output['choices']:
            return output['choices'][0]['message']['content'].strip()
        else:
            raise ValueError(f"Unexpected API response format: {output}")
    except Exception as e:
        raise RuntimeError(f"[Mistral Error] {e}")
