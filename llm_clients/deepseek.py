import os
import requests
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

API_URL = "https://api-inference.huggingface.co/models/deepseek-ai/deepseek-coder"

headers = {"Authorization": f"Bearer {HF_TOKEN}"}

def query(prompt: str) -> str:
    payload = {"inputs": prompt}
    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()
    output = response.json()
    if isinstance(output, list) and 'generated_text' in output[0]:
        return output[0]['generated_text'].strip()
    elif isinstance(output, dict) and 'generated_text' in output:
        return output['generated_text'].strip()
    else:
        return str(output)
