from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [
    "RAG improves accuracy of language models",
    "ChromaDB is used for vector storage",
    "Internships improve practical knowledge"
]

embeddings = model.encode(texts)

print("Number of embeddings:", len(embeddings))
print("Embedding vector size:", len(embeddings[0]))
