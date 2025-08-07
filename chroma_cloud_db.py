import uuid
from pypdf import PdfReader
import chromadb

# Step 1: Read PDF file using PyPDF
pdf_path = "./budget_speech.pdf"
reader = PdfReader(pdf_path)

documents = []
metadatas = []

for i, page in enumerate(reader.pages):
    text = page.extract_text()
    if text:  # Skip empty pages
        documents.append(text)
        metadatas.append({"page": i})

# Step 2: Initialize ChromaDB client
# client = chromadb.PersistentClient(path="./chroma_data")

client = chromadb.CloudClient(
  api_key='ck-9Ew8Ay6AkfGpHifMwYLN5Z2CCqJWjk8iNaPyDtzFQPw4',
  tenant='1e2f47df-e9ba-474a-ae4c-4f9631641ab4',
  database='dev_db'
)

# Step 3: Create or get the collection
collection = client.get_or_create_collection(name="budget_speech")

# Step 4: Add documents to the collection
collection.add(
    ids=[str(uuid.uuid4()) for _ in documents],
    documents=documents,
    metadatas=metadatas
)

print(f"✅ Ingested {len(documents)} pages from '{pdf_path}' into ChromaDB.")
