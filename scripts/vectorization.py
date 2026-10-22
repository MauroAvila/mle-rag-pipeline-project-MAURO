"""
Phase 2: Vectorization and Local Vector Store Setup
Module to load processed chunks, generate embeddings, and store them in a local Chroma vector database.
"""

import json
from pathlib import Path
from typing import Any, Dict, List
import chromadb
from sentence_transformers import SentenceTransformer


class VectorIndexer:
    """Handles text embedding generation and local vector store indexing for Northstar KB."""

    def __init__(
        self,
        input_path: str = "data/processed/processed_chunks.jsonl",
        db_path: str = "data/vector_store",
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.input_path = Path(input_path)
        self.db_path = Path(db_path)
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.client = chromadb.PersistentClient(path=str(self.db_path))
        self.collection = self.client.get_or_create_collection(name="northstar_support")

    def load_chunks(self) -> List[Dict[str, Any]]:
        """Loads processed chunks from the JSONL output file."""
        chunks = []
        if not self.input_path.exists():
            print(f"[Error] Processed chunks file not found at '{self.input_path}'.")
            return chunks

        with open(self.input_path, "r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    chunks.append(json.loads(line))
        return chunks

    def index_chunks(self) -> None:
        """Generates embeddings for chunks and indexes them into the local vector store."""
        chunks = self.load_chunks()
        if not chunks:
            print("[Warning] No chunks available to index.")
            return

        ids = [chunk["chunk_id"] for chunk in chunks]
        documents = [chunk["content"] for chunk in chunks]
        metadatas = [
            {
                "document_id": chunk["document_id"],
                "title": chunk["title"],
                "chunk_index": chunk["chunk_index"],
            }
            for chunk in chunks
        ]

        print(f"[Info] Generating embeddings for {len(chunks)} chunks using '{self.model_name}'...")
        embeddings = self.model.encode(documents, show_progress_bar=True).tolist()

        print(f"[Info] Storing vectors in local Chroma database at '{self.db_path}'...")
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        print(f"[Success] Successfully indexed {len(chunks)} chunks into the vector store.")


if __name__ == "__main__":
    print("--- Starting Phase 2: Vectorization and Indexing (Northstar) ---")
    indexer = VectorIndexer()
    indexer.index_chunks()