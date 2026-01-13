# Basics of Retrieval Augmented Generation (RAG)

## What is RAG?
RAG means:
- Retrieving relevant information from a knowledge base
- Giving it to a language model
- Generating a better answer

So formula:
Answer = LLM + Retrieved Knowledge

## Why RAG is Needed?
LLMs:
- may hallucinate
- forget recent data
- don’t know private/company data

RAG solves this.

## Main Components
1. User Query
2. Embed text into vectors
3. Store vectors in ChromaDB
4. Retrieve similar vectors
5. Pass them to LLM
6. Generate answer

## What is ChromaDB?
A vector database used to:
- store embeddings
- quickly search similar texts
- retrieve top k documents

## Simple Example
User asks: “What is RAG?”

Steps:
- Convert question into vector
- Search similar text in database
- Retrieve matching explanation
- Model uses it to answer

---
