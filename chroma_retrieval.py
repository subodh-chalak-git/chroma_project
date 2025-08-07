import json
import os
import openai
import streamlit as st
import chromadb
from dotenv import load_dotenv
load_dotenv()


with st.sidebar:
    st.title("Budget Speech Query")
    st.write("This app allows you to query the budget speech documents stored in ChromaDB.")
    OPENAI_API_KEY = st.text_input("OpenAI API Key", type="password")

if not OPENAI_API_KEY:
    st.error("Please provide a valid OpenAI API Key to proceed.")
    st.stop()


# gpt_client = openai.OpenAI(api_key=OPENAI_API_KEY)
# print(f"gpt_client: {gpt_client}")
# # collection_client = chromadb.PersistentClient('./chroma_data')
#
# collection_client = chromadb.CloudClient(
#   api_key=os.environ.get('CHROMA_API_KEY'),
#   tenant=os.environ.get("CHROMA_TENANT"),
#   database=os.environ.get("CHROMA_DB")
# )
#
# print(f"collection_client: {collection_client}")
#
# collection = collection_client.get_collection(name="budget_speech")
# print(f"collection: {collection}")


@st.cache_resource(show_spinner=False)
def initiate_data(api_key):
    gpt_client = openai.OpenAI(api_key=api_key)
    chroma_client = chromadb.CloudClient(
        api_key=os.environ.get('CHROMA_API_KEY'),
        tenant=os.environ.get("CHROMA_TENANT"),
        database=os.environ.get("CHROMA_DB")
    )
    collection = chroma_client.get_collection(name="budget_speech")
    return collection, gpt_client

collection, gpt_client = initiate_data(OPENAI_API_KEY)

query = st.text_input(label="Please enter your query")

if query:
    with st.spinner("Thinking..."):
        context = collection.query(
            query_texts=[query],
            n_results=10
        )['documents']

        # Combine multiple context documents into a single string
        combined_context = "\n\n".join([doc for doc_list in context for doc in doc_list])
        print(f"\ncombined_context: {combined_context}")

        prompt = f"{query}. Use this as context for answering: {combined_context}. If you are not aware or unclear, say you don't know. If context is not provides, say I don't know. If query is not related not document, do not answer it. Do not answer to anything which is not related to context provided."

        response = gpt_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt}
            ]
        )

        response_text = response.choices[0].message.content
        st.markdown(response_text)
else:
    st.info("Please enter your query")



