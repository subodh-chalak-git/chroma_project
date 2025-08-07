import chromadb
import uuid

client = chromadb.PersistentClient(path='./chroma_data')
collection = client.create_collection(name="policies")

with open("policies.txt", "r", encoding="utf-8") as f:
    policies: list['str'] = f.read().splitlines()

collection.add(
    ids = [str(uuid.uuid4()) for policy in policies],
    documents = policies,
    metadatas = [{"line": line} for line in range(len(policies))]
)

print(f"Colleciton count: {collection.count()}")


