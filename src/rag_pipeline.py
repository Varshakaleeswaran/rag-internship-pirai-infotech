from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings


# -------- Step 1: Text Chunking --------
def chunk_text(text, chunk_size=40, overlap=10):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap

    return chunks


# -------- Step 2: Build Persistent Vector Store --------
def build_vector_store(raw_documents):
    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Persistent ChromaDB
    client = chromadb.Client(
        Settings(persist_directory="chroma_db")
    )

    collection = client.get_or_create_collection(name="rag_collection")

    print("Chunking documents...")
    documents = []
    for doc in raw_documents:
        documents.extend(chunk_text(doc))

    print("Creating embeddings and storing in ChromaDB...")
    embeddings = model.encode(documents)

    collection.add(
        documents=documents,
        embeddings=embeddings.tolist(),
        ids=[str(i) for i in range(len(documents))]
    )

    return model, collection


# -------- Step 3: Retrieval --------
def retrieve_documents(model, collection, query, n_results=2):
    query_embedding = model.encode([query])
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=n_results
    )
    return results["documents"][0]


# -------- Step 4: Answer Generation --------
def generate_answer(query, retrieved_docs):
    print("\nGenerating answer using retrieved context...\n")

    context = " ".join(retrieved_docs)

    answer = f"""
Question:
{query}

Answer (Generated using retrieved knowledge):
{context}
"""
    return answer


# -------- Main Execution --------
if __name__ == "__main__":
    raw_documents = [
        "RAG stands for Retrieval Augmented Generation. It improves language model responses using retrieved external knowledge.",
        "ChromaDB is a vector database designed for storing and retrieving embeddings efficiently.",
        "Sentence Transformers convert text into numerical vector representations.",
        "RAG systems combine retrieval and generation to improve factual accuracy."
    ]

    model, collection = build_vector_store(raw_documents)

    query = "What is RAG?"
    print("\nUser Query:", query)

    retrieved_docs = retrieve_documents(model, collection, query)

    answer = generate_answer(query, retrieved_docs)
    print(answer)
