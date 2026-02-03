# rag_pipeline.py
# Day-2: Basic RAG Pipeline using ChromaDB

from sentence_transformers import SentenceTransformer
import chromadb

# 1. Sample documents (knowledge base)
documents = [
    "RAG stands for Retrieval Augmented Generation.",
    "ChromaDB is a vector database used for embeddings.",
    "Sentence transformers convert text into embeddings.",
    "RAG improves LLM responses using external knowledge."
]

# 2. Load embedding model
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# 3. Create ChromaDB client and collection
client = chromadb.Client()
collection = client.create_collection(name="rag_collection")

# 4. Generate embeddings and store them
print("Creating embeddings and storing in ChromaDB...")
embeddings = model.encode(documents)

collection.add(
    documents=documents,
    embeddings=embeddings.tolist(),
    ids=[str(i) for i in range(len(documents))]
)

# 5. User query
query = "What is RAG?"
print("\nUser Query:", query)

query_embedding = model.encode([query])

# 6. Retrieve relevant documents
results = collection.query(
    query_embeddings=query_embedding.tolist(),
    n_results=2
)

# 7. Display retrieved documents
print("\nRetrieved Documents:")
for doc in results["documents"][0]:
    print("-", doc)
