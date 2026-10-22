# Northstar Technical Support & Operations RAG Assistant: Project Summary

This document outlines the complete development lifecycle, architectural workflow, and Python scripts implemented to build the technical support and operations RAG (Retrieval-Augmented Generation) assistant for the educational platform **Northstar**.

---

## Project Overview & Guidelines

The project was constructed following a strict 5-phase modular architecture. All custom scripts were developed in Python and stored exclusively within the `scripts/` directory. To maintain a standardized codebase, all code, comments, variable names, and function names were written entirely in English.

---

## Summary of Written Python Scripts

### 1. `scripts/ingestion.py` (Phase 1: Ingestion, Cleaning & Chunking)

* **Purpose:** Scans the knowledge base directory, cleans raw Markdown documentation, and segments the text into manageable chunks.
* **Key Configuration:**
* **Chunk Size:** 500 words (balances contextual completeness with dense embedding limits).
* **Sliding Overlap:** 50 words (preserves boundary continuity and semantic flow).


* **Output:** Generates 10 processed chunks saved persistently to `data/processed/processed_chunks.jsonl` in JSONL format, complete with metadata (`chunk_id`, `document_id`, `title`, `chunk_index`).

### 2. `scripts/vectorization.py` (Phase 2: Vectorization & Local Vector Store)

* **Purpose:** Reads the processed chunks, generates high-quality vector embeddings, and indexes them into a local vector database.
* **Key Components:**
* Utilizes the `sentence-transformers` library with the lightweight open-source model `all-MiniLM-L6-v2`.
* Integrates **ChromaDB** using a persistent client (`PersistentClient`) to store vector embeddings, documents, and metadatas locally under `data/vector_store/`.



### 3. `scripts/retriever.py` (Phase 3: Retriever Configuration)

* **Purpose:** Configures the semantic search mechanism to query the local Chroma vector database.
* **Key Functionality:**
* Prompts the user interactively via the terminal with `"Write here your question:"`.
* Converts the input query into an embedding vector, performs a cosine/L2 similarity search, and retrieves the top relevant chunks (`top_k=2`) along with their metadata and distance relevance scores.



### 4. `scripts/generator.py` (Phase 4: RAG Generator Integration)

* **Purpose:** Combines semantic retrieval with a generative Large Language Model (LLM) to produce natural, grounded support answers.
* **Key Components:**
* Utilizes the **Mistral API** (`ministral-14b-latest`) configured via an OpenAI-compatible client endpoint (`[https://api.mistral.ai/v1](https://api.mistral.ai/v1)`) using credentials from `.env`.
* Enforces a strict system prompt instructing the LLM to answer *only* using the provided context, safely declining to answer or hallucinate when knowledge is missing from the database.



### 5. `scripts/rag_pipeline.py` (Phase 5: End-to-End Pipeline & Interactive Assistant)

* **Purpose:** Acts as the master orchestrator uniting all components into a unified, continuous technical support application.
* **Key Functionality:**
* Initializes the embedding model, vector store collection, and Mistral client upon startup.
* Runs a continuous interactive loop allowing support operators to type sequential queries in English, retrieve context, and receive accurate, context-grounded troubleshooting responses in real time.