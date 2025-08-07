import chromadb
import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# client = chromadb.HttpClient(host="localhost", port=8000)
client = chromadb.PersistentClient(path='./chroma_data')
print(f"My Client: {client}")


collection = client.get_or_create_collection(name="budget_speech")
print(f"collection: {collection}")
print(f"Collection count: {collection.count()}")


with st.sidebar:
    st.title("Budget Speech Query")
    st.write("This app allows you to query the budget speech documents stored in ChromaDB.")
    OPENAI_API_KEY = st.text_input("OpenAI API Key", type="password")

if not OPENAI_API_KEY:
    st.error("Please provide a valid OpenAI API Key to proceed.")

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

vector_store = Chroma(
    collection_name='budget_speech',
    embedding_function=embeddings,
    persist_directory="./chroma_data"
)


query = st.text_input(label="Please enter your budget query")

results = collection.query(
    query_texts=[query],
    n_results = 5
)

print(f"Results: {results}")

for index, result in enumerate(results['documents']):
    print(f"Query: {index}")
    print(f"Result: {result}")


