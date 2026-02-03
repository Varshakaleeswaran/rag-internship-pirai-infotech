from sentence_transformers import SentenceTransformer
import chromadb

def build_vector_store(documents):
    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    client = chromadb.Client()
    collection = client.create_collection(name="rag_collection")

    print("Creating embeddings and storing in ChromaDB...")
    embeddings = model.encode(documents)

    collection.add(
        documents=documents,
        embeddings=embeddings.tolist(),
        ids=[str(i) for i in range(len(documents))]
    )

    return model, collection


def retrieve_documents(model, collection, query, n_results=2):
    query_embedding = model.encode([query])
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=n_results
    )
    return results["documents"][0]


if __name__ == "__main__":
    documents = [
        "RAG stands for Retrieval Augmented Generation.",
        "ChromaDB is a vector database used for embeddings.",
        "Sentence transformers convert text into embeddings.",
        "RAG improves LLM responses using external knowledge."
    ]

    model, collection = build_vector_store(documents)

    query = "What is RAG?"
    print("\nUser Query:", query)

    retrieved_docs = retrieve_documents(model, collection, query)

    print("\nRetrieved Documents:")
    for doc in retrieved_docs:
        print("-", doc)
