import chromadb

chroma_client = chromadb.Client()

collection = chroma_client.get_or_create_collection("test_collection")

documents = [
    {"id": "doc1", "text": "In 2030 the cure of cancer will be discovered."},
    {"id": "doc2", "text": "In 2033 The impact of climate change on global agriculture will be significant."},
    {"id": "doc3", "text": "In 2100 Artificial intelligence will revolutionize the healthcare industry."}
]

for doc in documents:
    collection.upsert(ids=[doc["id"]], documents=[doc["text"]])

query = "since when AI will revolutionize any industry?"

results = collection.query(
    query_texts=[query],
    n_results=2
)

print(f"Query: {query}\n\n")

for i, doc in enumerate(results["documents"][0]):
    doc_id = results["ids"][0][i]
    distance = results["distances"][0][i]
    print(f"Document ID: {doc_id}, Distance: {distance}, Text: {doc}")
