import streamlit as st
import pandas as pd
from llm_clients.gemini import query as extract_with_gemini

# --- Pre-defined prompts based on CSV data ---
PROMPT_TEMPLATES = {
    "running shoes": [
        "What are the best running shoes for marathon training?", "Recommend a durable running shoe for under $120.",
        "I have flat feet, what are the top stability running shoes?", "Show me lightweight running shoes for racing.",
        "Best running shoes for beginners.", "Compare Nike and Hoka running shoes.", "What are some good trail running shoes?",
        "Most comfortable running shoes for everyday wear."
    ],
    "laptops": [
        "What is the best budget laptop for a college student?", "Recommend a lightweight laptop with long battery life.",
        "Best gaming laptop under $1500.", "MacBook Air vs Dell XPS for programming.", "Show me 2-in-1 convertible laptops.",
        "Laptops with the best keyboards for writers.", "Most reliable laptop brands for 2024.", "Best laptops for video editing on a budget."
    ]
}

# --- This prompt is for our first layer of analysis: extracting names ---
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

# --- Prompt for the second layer of analysis ---
KEY_THEMES_PROMPT_TEMPLATE = """
You are a market research analyst. Your task is to analyze the following collection of AI-generated product recommendations.
Identify and summarize the most common "buying factors," recurring product attributes, or buzzwords that the AI uses to justify its recommendations.

Rules:
- Focus on the *reasons* given for a recommendation (e.g., "battery life", "lightweight", "durable", "value for money", "gentle formula").
- Group similar concepts together.
- Present your findings as a concise summary using markdown formatting with bullet points.

Here is the collection of texts to analyze:
\"\"\"{all_responses}\"\"\"
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

# --- This function will perform our meta-analysis ---
def summarize_key_themes(all_responses_text: str) -> str:
    """Uses an LLM to analyze all responses and extract key themes."""
    try:
        prompt = KEY_THEMES_PROMPT_TEMPLATE.format(all_responses=all_responses_text)
        summary = extract_with_gemini(prompt)
        return summary
    except Exception as e:
        print(f"[Summarizer Error] Gemini failed to summarize: {e}")
        return "Could not generate insights summary due to an error."

def run_geo_analysis(category: str, get_llm_response_func):
    if category.lower() not in PROMPT_TEMPLATES:
        return None, None, [], f"Category '{category}' not found. Please try one of: {list(PROMPT_TEMPLATES.keys())}"

    prompts = PROMPT_TEMPLATES[category.lower()]
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
        return None, None, raw_responses, "Analysis complete, but no recognizable brands or products were extracted."

    # --- Call the summarizer function after collecting all data ---
    progress_bar.progress(1.0, text="Generating insights summary...")
    all_responses_joined = "\n\n---\n\n".join([item["response"] for item in raw_responses])
    key_themes_summary = summarize_key_themes(all_responses_joined)

    brand_counts = pd.Series(all_brands).value_counts().rename("Mention Count")
    product_counts = pd.Series(all_products).value_counts().rename("Mention Count")
    
    analysis_results = {"brands": brand_counts, "products": product_counts}
    
    # Return the new summary along with the other data
    return analysis_results, key_themes_summary, raw_responses, "Analysis complete."
