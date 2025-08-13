import streamlit as st
import pandas as pd
from llm_clients.gemini import query as extract_with_gemini

# --- Using the prompts from the CSV file as our pre-defined set ---
PROMPT_TEMPLATES = {
    "Tech": [
        "What are the best gaming mouse brands with models available for under $100?",
        "Best laptop for office work under $1000",
        "What keyboard should I buy?",
    ],
    "Skincare": [
        "Which brand is best for anti-aging night creams with retinol?",
        "Best fragrance-free moisturizers for sensitive skin in humid weather",
        "Can you give me some suggestions for natural skincare routines?",
    ],
    "Fitness": [
        "Where can I buy home fitness equipment for small spaces?",
        "Best at-home cardio equipment for beginners",
        "Suggestion for a gym near me",
    ],
    "Home Essentials": [
        "Which brand is best for a durable and easy-to-use knife sharpener?",
        "Best air purifiers under $100 for bedrooms?",
        "What are some suggestions for organizing a small kitchen efficiently?",
    ],
    "Fashion": [
        "Which brand has the best formal wear?",
        "Best affordable fashion brands for students",
        "Suggestion for clothes to buy",
    ],
    # Add more categories and prompts as needed
}

EXTRACTION_PROMPT_TEMPLATE = """
Analyze the following text. Extract brand names and specific product names mentioned.

Rules:
- List only real-world brands and products.
- If no brands are mentioned, write "None".
- If no products are mentioned, write "None".

Output format (include nothing else but the lists):
Brands:
- [Brand Name 1]

Products:
- [Product Name 1]

Here is the text:
\"\"\"{response}\"\"\"
"""

def parse_extraction(extraction_text: str) -> tuple[list, list]:
    brands, products = [], []
    current_section = None
    for line in extraction_text.splitlines():
        if "brands:" in line.lower(): current_section = "brands"
        elif "products:" in line.lower(): current_section = "products"
        elif line.startswith("- ") and "none" not in line.lower():
            item = line[2:].strip()
            if current_section == "brands": brands.append(item)
            elif current_section == "products": products.append(item)
    return brands, products

def run_geo_analysis(category: str, get_llm_response_func):
    if category not in PROMPT_TEMPLATES:
        return None, [], f"Category '{category}' not found. Please try one of: {list(PROMPT_TEMPLATES.keys())}"

    prompts = PROMPT_TEMPLATES[category]
    raw_responses = []
    all_brands, all_products = [], []

    progress_bar = st.progress(0, text="Starting analysis...")
    for i, prompt in enumerate(prompts):
        try:
            response = get_llm_response_func(prompt)
            raw_responses.append({"prompt": prompt, "response": response})
            
            extraction_prompt = EXTRACTION_PROMPT_TEMPLATE.format(response=response)
            extraction_text = extract_with_gemini(extraction_prompt)
            
            brands, products = parse_extraction(extraction_text)
            all_brands.extend(brands)
            all_products.extend(products)
        except Exception as e:
            st.warning(f"Error on prompt: '{prompt}'. Skipping. Details: {e}")
            continue
        
        progress_bar.progress((i + 1) / len(prompts), text=f"Processing prompt {i+1}/{len(prompts)}")

    if not all_brands and not all_products:
        return None, raw_responses, "Analysis complete, but no recognizable brands or products were extracted."

    brand_counts = pd.Series(all_brands).value_counts().rename("Mention Count")
    product_counts = pd.Series(all_products).value_counts().rename("Mention Count")
    
    return {"brands": brand_counts, "products": product_counts}, raw_responses, "Analysis complete."
