import streamlit as st

# --- Importing each "query" function with a unique alias ---
from llm_clients.chatgpt import query as get_gpt_response
from llm_clients.gemini import query as get_gemini_response
from llm_clients.deepseek import query as get_deepseek_response

# ======================================================================================
# --- Page Configuration & Header ---
# ======================================================================================

st.set_page_config(
    page_title="Prompty",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)


st.title("Prompty")

# ======================================================================================
# --- Sidebar Controls ---
# ======================================================================================

with st.sidebar:
    st.image("logo.jpg", width=250)
    
    st.header("Controls")
    model_option = st.selectbox(
        "Choose a Model:",
        ("GPT-3.5-Turbo", "Gemini", "Mistral-7B Instruct"),
        key="model_select"
    )
    st.info("Your chat history will be cleared if you switch models or refresh the page.")

# ======================================================================================
# --- Chat History Management ---
# ======================================================================================

if "current_model" not in st.session_state or st.session_state.current_model != model_option:
    st.session_state.current_model = model_option
    st.session_state.messages = []

# Display past messages from history on rerun.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ======================================================================================
# --- Chat Input & Response Logic ---
# ======================================================================================

if prompt := st.chat_input("Ask your question here..."):

    # Display user's message and add to history.
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- Call the appropriate backend function ---
    with st.chat_message("assistant"):
        with st.spinner(f"Thinking with {model_option}..."):
            try:
                response = ""
                # Call the correct aliased function based on user selection.
                if model_option == "GPT-3.5-Turbo":
                    response = get_gpt_response(prompt)
                elif model_option == "Gemini":
                    response = get_gemini_response(prompt)
                elif model_option == "Mistral-7B Instruct":
                    response = get_deepseek_response(prompt)
                
                # Display the successful response.
                st.markdown(response)
                # Save the successful response to history.
                st.session_state.messages.append({"role": "assistant", "content": response})

            except Exception as e:
                # --- Error Handling ---
                error_message = str(e)
                st.error(error_message)
                # Save the error to history so the user knows what happened.
                st.session_state.messages.append({"role": "assistant", "content": error_message})
