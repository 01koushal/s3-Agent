import os
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

# Load environment variables
MONGO_URI = os.getenv("MONGO_URI")

# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client["rag_db"]
collection = db["data-for-ai"]

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Sample data (replace with your actual data source)
documents = [
    {"text": "Artificial Intelligence is the simulation of human intelligence."},
    {"text": "Machine Learning is a subset of AI."},
    {"text": "RAG combines retrieval and generation for better answers."}
]

def create_embedding(text):
    return model.encode(text).tolist()

def store_documents(docs):
    for doc in docs:
        embedding = create_embedding(doc["text"])
        collection.insert_one({
            "text": doc["text"],
            "embedding": embedding
        })

if __name__ == "__main__":
    store_documents(documents)
    print("Data inserted with embeddings")
