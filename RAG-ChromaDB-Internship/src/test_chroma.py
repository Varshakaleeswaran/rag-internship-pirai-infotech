import chromadb

# create a simple in-memory Chroma client
client = chromadb.Client()

# create a test collection
collection = client.create_collection(name="test_collection")

# add some sample data
collection.add(
    ids=["1", "2"],
    documents=["RAG uses retrieval and generation", "ChromaDB is a vector database"]
)

# simple search query
results = collection.query(
    query_texts=["What is ChromaDB?"],
    n_results=2
)

print("ChromaDB working successfully!")
print(results)
