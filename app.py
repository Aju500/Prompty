import streamlit as st
import pandas as pd

# --- Importing our backend toolkits ---
from llm_clients.chatgpt import query as get_gpt_response
from llm_clients.gemini import query as get_gemini_response
from llm_clients.deepseek import query as get_mistral_response
from prompt_processer import run_geo_analysis, PROMPT_TEMPLATES

# ======================================================================================
# --- Page Configuration & CSS ---
# ======================================================================================
st.set_page_config(page_title="Prompty GEO Monitor", page_icon="logo.png", layout="wide")


st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    :root {
        --primary-color: #3A86FF; --background-color: #0A0F1E; --secondary-background-color: #10204A;
        --sidebar-background-color: #0D162F; --text-color: #FFFFFF; --font-family: 'Inter', sans-serif;
    }
    body, h1, h2, h3 { font-family: var(--font-family); color: var(--text-color); }
    [data-testid="stAppViewContainer"] { background-color: var(--background-color); }
    [data-testid="stSidebar"] { background-color: var(--sidebar-background-color); border-right: 1px solid #1A2C5A; }
    .stButton > button { border-radius: 0.5rem; background-color: var(--primary-color); color: var(--text-color); border: none; }
    [data-testid="stSelectbox"], [data-testid="stTextInput"] { background-color: var(--secondary-background-color); }
</style>
""", unsafe_allow_html=True)

# ======================================================================================
# --- Sidebar ---
# ======================================================================================
with st.sidebar:
    st.image("logo.png", use_container_width=True)
    st.header("Analysis Controls")
    model_option = st.selectbox(
        "Choose Model for Analysis:",
        ("GPT-3.5-Turbo", "Gemini", "Mistral-7B Instruct")
    )
    category = st.selectbox(
        "Choose a Product Category:",
        options=list(PROMPT_TEMPLATES.keys())
    )
    st.info("This tool runs a batch of pre-defined prompts for a category to analyze brand and product mentions in LLM responses.")

# ======================================================================================
# --- Main Application UI ---
# ======================================================================================
st.title("Prompty: Generative Engine Optimization Monitor")
st.write(f"Analyzing **{category}** prompts using **{model_option}**.")

# --- Map model selection to the correct function ---
llm_functions = {
    "GPT-3.5-Turbo": get_gpt_response,
    "Gemini": get_gemini_response,
    "Mistral-7B Instruct": get_mistral_response
}
selected_llm_func = llm_functions[model_option]

# --- Run Analysis Button ---
if st.button("Run GEO Analysis 🚀 ", use_container_width=True):
    # --- NEW: Unpack the new summary variable from the function call ---
    analysis, summary, raw_data, message = run_geo_analysis(category, selected_llm_func)
    
    if analysis:
        st.success(message)
        st.header("GEO Analysis Dashboard 📊")

        # --- NEW: Display the Key Insights Summary Section ---
        st.subheader("Key Buying Factors & Buzzwords 💡")
        st.markdown(summary)
        st.divider()

        # --- The original data tables and charts ---
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Top Brand Mentions")
            st.dataframe(analysis["brands"])
            st.bar_chart(analysis["brands"])
        with col2:
            st.subheader("Top Product Mentions")
            st.dataframe(analysis["products"])
            st.bar_chart(analysis["products"])

        with st.expander("View Raw Prompts & Responses"):
            st.json(raw_data)
    else:
        st.error(message)
        if raw_data:
            with st.expander("View Raw Prompts & Responses"):
                st.json(raw_data)
