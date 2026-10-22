"""
Phase 3: Retriever Configuration
Module to query the local Chroma vector store and retrieve relevant knowledge base chunks for user queries.
"""

from pathlib import Path
from typing import Any, Dict, List
import chromadb
from sentence_transformers import SentenceTransformer


class KnowledgeBaseRetriever:
    """Handles semantic search and chunk retrieval from the local Chroma vector database."""

    def __init__(
        self,
        db_path: str = "data/vector_store",
        model_name: str = "all-MiniLM-L6-v2",
        collection_name: str = "northstar_support",
    ) -> None:
        self.db_path = Path(db_path)
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.client = chromadb.PersistentClient(path=str(self.db_path))
        self.collection = self.client.get_collection(name=collection_name)

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Performs a semantic similarity search against the vector database."""
        # Generate embedding for the input query
        query_embedding = self.model.encode([query]).tolist()

        # Query the Chroma collection
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )

        # Format the retrieved results
        formatted_results = []
        if results and "documents" in results and results["documents"]:
            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0] if "distances" in results else [None] * len(documents)
            ids = results["ids"][0]

            for chunk_id, doc, meta, dist in zip(ids, documents, metadatas, distances):
                formatted_results.append({
                    "chunk_id": chunk_id,
                    "content": doc,
                    "metadata": meta,
                    "distance": dist
                })

        return formatted_results


if __name__ == "__main__":
    print("--- Starting Phase 3: Retriever Testing (Northstar) ---")
    retriever = KnowledgeBaseRetriever()
    
    # Receive query interactively from the user
    user_query = input("Write here your question: ")
    
    if user_query.strip():
        retrieved_chunks = retriever.search(user_query, top_k=2)
        
        print(f"\n[Result] Retrieved {len(retrieved_chunks)} relevant chunks:")
        for index, res in enumerate(retrieved_chunks, start=1):
            print(f"\n--- Result #{index} ---")
            print(f"  • Chunk ID: {res['chunk_id']}")
            print(f"  • Title: {res['metadata'].get('title')}")
            print(f"  • Distance Score: {res['distance']}")
            print(f"  • Content Preview: {res['content'][:150]}...")
    else:
        print("[Warning] No question was provided.")