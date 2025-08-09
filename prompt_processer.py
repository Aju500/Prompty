import pandas as pd 
import os
from dotenv import load_dotenv
from llm_clients import gemini, chatgpt, deepseek
import time
import json

load_dotenv()

EXTRACTION_PROMPT_TEMPLATE = """
Extract and return:
1. A list of product names mentioned in this text, double check if each is a real world product that belongs to a brand before returning, double check that each brand is real and current before returning, and return just the word "none" under the beginning of the list if there are none.
2. A list of brand names mentioned in this text, double check that each brand is real and current before returning, and return just the word "none" under the beginning of the list if there are none.

Output format (include nothing else):
Products:
- [product 1]
- [product 2]
...

Brands:
- [brand 1]
- [brand 2]
...

Here is the text:
\"\"\"{response}\"\"\"
"""

def extract_mentions(response: str) -> tuple[list[str], list[str]]:
    try:
        prompt = EXTRACTION_PROMPT_TEMPLATE.format(response=response)
        extraction_response = gemini.query(prompt)
    except Exception as e:
        print(f"[Extractor error] Gemini failed to extract: {e}")
        return [], []

    products, brands = [], []
    current = None
    for line in extraction_response.splitlines():
        line = line.strip()
        if line.lower().startswith("products"):
            current = "product"
        elif line.lower().startswith("brands"):
            current = "brand"
        elif line.startswith("- "):
            item = line[2:].strip()
            if current == "product":
                products.append(item)
            elif current == "brand":
                brands.append(item)
    return products, brands


def get_llm_client(name: str):
    name = name.lower()
    if name == "gemini":
        return gemini
    elif name == "chatgpt":
        return chatgpt
    elif name == "deepseek":
        return deepseek
    else:
        return gemini  # default fallback


def process_prompts(csv_path="prompts+data2.csv"):
    df = pd.read_csv(csv_path, dtype={"productMentions": str, "brandMentions": str, "llm": str, "fullResponse": str})

    updated_rows = 0

    for i, row in df.iterrows():
        if pd.notna(row["productMentions"]) and pd.notna(row["brandMentions"]) and pd.notna(row["fullResponse"]):
            continue

        prompt = row["prompt"]
        llm_name = row.get("llm", "gemini")
        llm_client = get_llm_client(llm_name)

        try:
            print(f"[{llm_name}] Querying for: {prompt}")
            response = llm_client.query(prompt)
        except Exception as e:
            print(f"[Error] {llm_name} failed for row {i}: {e}")
            try:
                print(f"[Fallback] Using Gemini for: {prompt}")
                response = gemini.query(prompt)
            except Exception as fallback_error:
                print(f"[Fallback Error] Gemini also failed: {fallback_error}")
                continue  # Skip this row entirely

        products, brands = extract_mentions(response)
        df.at[i, "productMentions"] = "; ".join(products)
        df.at[i, "brandMentions"] = "; ".join(brands)
        df.at[i, "fullResponse"] = json.dumps(response)
        updated_rows += 1

        time.sleep(1)

    if updated_rows > 0:
        df.to_csv(csv_path, index=False)
        print(f"✅ Updated {updated_rows} rows.")
    else:
        print("No new rows updated.")

if __name__ == "__main__":
    process_prompts()
