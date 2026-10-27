"""
Phase 2: Vectorization, Local Vector Database Indexing, Metrics, and Persistent Reporting
Module to generate embeddings and store chunks into ChromaDB for the Northstar RAG pipeline.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import chromadb
from sentence_transformers import SentenceTransformer


def process_vectorization(
    chunks_path: str = "data/processed/processed_chunks.jsonl",
    db_path: str = "data/vector_store",
    report_path: str = "data/processed/vectorization_report.json",
    model_name: str = "all-MiniLM-L6-v2",
    collection_name: str = "northstar_support"
) -> None:
    """Reads processed chunks, embeds them, indexes into ChromaDB, and generates a persistent report."""
    chunks_file = Path(chunks_path)
    db_dir = Path(db_path)
    report_file = Path(report_path)
    
    report_file.parent.mkdir(parents=True, exist_ok=True)

    if not chunks_file.exists():
        print(f"[Error] Processed chunks file not found at {chunks_path}. Run Phase 1 ingestion first.")
        return

    # Load chunks from JSONL
    chunks = []
    with open(chunks_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line.strip()))

    total_chunks = len(chunks)
    print(f"[Info] Loaded {total_chunks} chunk(s) from {chunks_path}")

    if total_chunks == 0:
        print("[Warning] No chunks found to vectorize.")
        return

    # Initialize embedding model and vector database
    print(f"[Info] Initializing embedding model: {model_name}")
    encoder = SentenceTransformer(model_name)
    
    # Validate embedding dimension
    sample_embedding = encoder.encode("Test embedding dimension")
    embedding_dim = len(sample_embedding)

    print(f"[Info] Connecting to ChromaDB persistent store at {db_path}")
    db_client = chromadb.PersistentClient(path=str(db_dir))
    
    # Create or get collection
    collection = db_client.get_or_create_collection(name=collection_name)

    # Prepare data for ChromaDB upsert
    documents = [c["content"] for c in chunks]
    metadatas = [{
        "document_id": c["document_id"],
        "title": c["title"],
        "chunk_index": c["chunk_index"]
    } for c in chunks]
    ids = [c["chunk_id"] for c in chunks]
    
    print("[Info] Generating embeddings and indexing into ChromaDB...")
    embeddings = encoder.encode(documents).tolist()

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )

    final_count = collection.count()
    print(f"[Success] Successfully indexed {final_count} chunks into ChromaDB collection '{collection_name}'")

    # --- Calculate Vectorization Metrics ---
    metrics_data = {
        "timestamp": datetime.now().isoformat(),
        "model_name": model_name,
        "embedding_dimension": embedding_dim,
        "collection_name": collection_name,
        "total_chunks_loaded": total_chunks,
        "total_chunks_indexed": final_count,
        "synchronization_status": final_count == total_chunks
    }

    # Save report persistently to JSON file
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=4, ensure_ascii=False)

    print(f"[Success] Persistent vectorization report saved successfully to {report_path}")

    # --- Print Console Summary Report ---
    print("\n==============================================")
    print("         VECTORIZATION METRICS REPORT         ")
    print("==============================================")
    print(f"• Model Used                : {model_name}")
    print(f"• Embedding Dimension       : {embedding_dim}")
    print(f"• Collection Name           : {collection_name}")
    print(f"• Total Chunks Indexed      : {final_count}")
    print(f"• Synchronization Status    : {'OK (Matched)' if metrics_data['synchronization_status'] else 'MISMATCH'}")
    print("==============================================\n")


if __name__ == "__main__":
    print("--- Starting Phase 2: Vectorization & Database Indexing (Northstar) ---")
    process_vectorization()