# rag_pipeline.py
# Day-4: RAG Pipeline with External Documents (TXT files)

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import os


# -------- Step 0: Load Documents from Folder --------
def load_documents_from_folder(folder_path):
    documents = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)

            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()
                documents.append(text)

    return documents


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


# -------- Step 4: Simple Answer Generation --------
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

    #  Load real documents from data folder
    raw_documents = load_documents_from_folder("data")

    model, collection = build_vector_store(raw_documents)

    print("\n RAG System Ready! Type 'exit' to stop.\n")

    # -------- Interactive Loop --------
    while True:
        query = input("Enter your question: ")

        if query.lower() in ["exit", "quit", "stop"]:
            print("Exiting RAG system...")
            break

        retrieved_docs = retrieve_documents(model, collection, query)

        answer = generate_answer(query, retrieved_docs)

        print(answer)
        print("-" * 60)

