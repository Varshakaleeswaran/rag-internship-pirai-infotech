import chromadb

client = chromadb.Client()

collection = client.create_collection(name="day2_rag_demo")

collection.add(
    documents=[
        "Retrieval Augmented Generation improves LLM accuracy",
        "ChromaDB is a vector database",
        "Embeddings represent semantic meaning of text"
    ],
    ids=["doc1", "doc2", "doc3"]
)

results = collection.query(
    query_texts=["What improves LLM accuracy?"],
    n_results=2
)

print("Retrieved documents:")
print(results["documents"])
