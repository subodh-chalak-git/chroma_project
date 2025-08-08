__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import openai
import streamlit as st
import chromadb
from dotenv import load_dotenv
from langsmith.wrappers import wrap_openai
from langsmith import traceable

# Load environment variables
load_dotenv()

# Streamlit page config
st.set_page_config(page_title="Budget Speech Query", layout="wide")

# --- Initialize session state ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "show_history" not in st.session_state:
    st.session_state.show_history = True

if "last_response" not in st.session_state:
    st.session_state.last_response = None

if "last_context" not in st.session_state:
    st.session_state.last_context = None  # store ChromaDB context per turn

# --- Sidebar UI ---
with st.sidebar:
    st.title("💬 Budget Speech Query")
    st.write("Query budget speech documents stored in ChromaDB.")
    OPENAI_API_KEY = st.text_input("🔑 OpenAI API Key", type="password")

    if st.button("🧾 Toggle Chat History"):
        st.session_state.show_history = not st.session_state.show_history

    if st.button("🧹 Clear Chat History"):
        st.session_state.messages = []
        st.session_state.last_response = None
        st.session_state.last_context = None

# --- Require OpenAI API Key ---
if not OPENAI_API_KEY:
    st.error("Please provide a valid OpenAI API Key to proceed.")
    st.stop()

# --- Initialize ChromaDB and OpenAI client ---
@st.cache_resource(show_spinner=False)
def initiate_data(api_key):
    print("✅ Initializing GPT and ChromaDB clients...")
    gpt_client = wrap_openai(openai.OpenAI(api_key=api_key))
    chroma_client = chromadb.CloudClient(
        api_key=os.environ.get('CHROMA_API_KEY'),
        tenant=os.environ.get("CHROMA_TENANT"),
        database=os.environ.get("CHROMA_DB")
    )
    collection = chroma_client.get_collection(name="budget_speech")
    return collection, gpt_client


@traceable(name="Query with Context + GPT")
def ask_query_with_trace(messages, gpt_client):
    print("Calling langsmith traceable...")
    # OpenAI call
    response = gpt_client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    return response.choices[0].message.content


try:
    collection, gpt_client = initiate_data(OPENAI_API_KEY)
except:
    st.markdown("### 🤖 Error")
    st.markdown("Something went wrong. Please try again later.")
    st.stop()

# --- Show chat history (collapsed by default) ---
if st.session_state.show_history and st.session_state.messages:
    with st.expander("📜 Chat History", expanded=False):
        for i, msg in enumerate(st.session_state.messages):
            role = "🧑 You" if msg["role"] == "user" else "🤖 Assistant"
            st.markdown(f"**{role}:** {msg['content']}")
            if i < len(st.session_state.messages) - 1:
                st.markdown("---------------------------")  # Divider between messages


# --- Query Input Form ---
st.markdown("## 📝 Ask about the Budget Speech")

with st.form("query_form"):
    query = st.text_input("Enter your query", placeholder="e.g., What are the key tax reforms?")
    submitted = st.form_submit_button("Submit")

# --- On Submit ---
if submitted and query and collection and gpt_client:
    try:
        with st.spinner("🤔 Thinking..."):
            # Fetch vector context from ChromaDB
            context = collection.query(query_texts=[query], n_results=5)['documents']
            combined_context = "\n\n".join([doc for doc_list in context for doc in doc_list])
            st.session_state.last_context = combined_context  # Save for debugging or auditing

            # Build chat messages
            messages = []

            # Add a fresh system message each time with current context
            system_prompt = (
                "You are a helpful assistant who answers queries about Indian Government budget speeches."
                "While answering the question, use the bullet points, emojis, symbols to make it look attractive for the end user to read."
                "Use the following context to answer. If the question is unrelated, say you don't know.\n\n"
                f"Context:\n{combined_context}"
            )
            messages.append({"role": "system", "content": system_prompt})

            # Add chat history from previous turns
            messages.extend(st.session_state.messages)

            # Add current user query
            messages.append({"role": "user", "content": query})

            # call ask_query_wih_trace function to run the query
            response_text = ask_query_with_trace(messages=messages, gpt_client=gpt_client)

            # Save conversation turn
            st.session_state.messages.append({"role": "user", "content": query})
            st.session_state.messages.append({"role": "assistant", "content": response_text})
            st.session_state.last_response = response_text
    except:
        st.markdown("### 🤖 Error")
        st.markdown("Something went wrong. Please try again later.")

# --- Display assistant response even after rerun ---
if st.session_state.last_response:
    st.markdown("### 🤖 Assistant Response")
    st.markdown(st.session_state.last_response)
