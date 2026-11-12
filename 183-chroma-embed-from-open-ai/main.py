import chromadb
import os

from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")

openai_ef = embedding_functions.OpenAIEmbeddingFunction(api_key=openai_key, model_name="text-embedding-3-small")
default_ef = embedding_functions.DefaultEmbeddingFunction()

client = chromadb.PersistentClient(path="./183-chroma-embed-from-open-ai/chroma_persisted")
collection = client.get_or_create_collection("test_persisted_collection", embedding_function=openai_ef)

documents = [
    {"id": "doc1", "text": "Hello, world!"},
    {"id": "doc2", "text": "How are you today?"},
    {"id": "doc3", "text": "Goodbye, see you later!"},
    {
        "id": "doc4",
        "text": "Microsoft is a technology company that develops software. It was founded by Bill Gates and Paul Allen in 1975.",
    },
    {
        "id": "doc5",
        "text": "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think like humans and mimic their actions.",
    },
    {
        "id": "doc6",
        "text": "Machine Learning (ML) is a subset of AI that focuses on the development of algorithms that allow computers to learn from and make predictions based on data.",
    },
    {
        "id": "doc7",
        "text": "Deep Learning is a subset of Machine Learning that uses neural networks with many layers to analyze various factors of data.",
    },
    {
        "id": "doc8",
        "text": "Natural Language Processing (NLP) is a branch of AI that helps computers understand, interpret, and respond to human language.",
    },
    {
        "id": "doc9",
        "text": "AI can be categorized into two types: Narrow AI, which is designed to perform a narrow task, and General AI, which can perform any intellectual task that a human can do.",
    },
    {
        "id": "doc10",
        "text": "Computer Vision is a field of AI that enables computers to interpret and make decisions based on visual data from the world.",
    },
    {
        "id": "doc11",
        "text": "Reinforcement Learning is an area of Machine Learning where an agent learns to make decisions by taking actions in an environment to achieve maximum cumulative reward.",
    },
    {
        "id": "doc12",
        "text": "The Turing Test, proposed by Alan Turing, is a measure of a machine's ability to exhibit intelligent behavior equivalent to, or indistinguishable from, that of a human.",
    },
]

for doc in documents:
    collection.upsert(ids=[doc["id"]], documents=[doc["text"]])


query = "find document related to technology company"

results = collection.query(
    query_texts=[query],
    n_results=2
)

print(f"Query: {query}\n\n")

for i, doc in enumerate(results["documents"][0]):
    doc_id = results["ids"][0][i]
    distance = results["distances"][0][i]
    print(f"Document ID: {doc_id}, Distance: {distance}, Text: {doc}")
