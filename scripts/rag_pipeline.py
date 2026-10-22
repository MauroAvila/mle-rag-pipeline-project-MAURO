"""
Phase 5: End-to-End RAG Pipeline & Interactive Assistant
Module to run the complete Northstar support and operations RAG assistant.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL_NAME = "ministral-14b-latest"
MISTRAL_BASE_URL = "https://api.mistral.ai/v1"


def get_client() -> OpenAI:
    """Initializes and returns the OpenAI client configured for Mistral API."""
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        print("Error: MISTRAL_API_KEY not found in environment variables or .env file.")
        sys.exit(1)
    return OpenAI(api_key=api_key, base_url=MISTRAL_BASE_URL)


class NorthstarRAGAssistant:
    """End-to-end RAG assistant combining vector search and generative grounded responses."""

    def __init__(
        self,
        db_path: str = "data/vector_store",
        model_name: str = "all-MiniLM-L6-v2",
        collection_name: str = "northstar_support",
    ) -> None:
        self.db_path = Path(db_path)
        self.model_name = model_name
        
        print("[Info] Initializing embedding model and vector database...")
        self.encoder = SentenceTransformer(model_name)
        self.db_client = chromadb.PersistentClient(path=str(self.db_path))
        self.collection = self.db_client.get_collection(name=collection_name)
        
        print("[Info] Connecting to Mistral generative model...")
        self.openai_client = get_client()

    def retrieve_context(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieves relevant chunks from the vector database based on semantic similarity."""
        query_embeddings = self.encoder.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_embeddings,
            n_results=top_k
        )

        formatted_results = []
        if results and "documents" in results and results["documents"]:
            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            ids = results["ids"][0]

            for chunk_id, doc, meta in zip(ids, documents, metadatas):
                formatted_results.append({
                    "chunk_id": chunk_id,
                    "content": doc,
                    "metadata": meta
                })

        return formatted_results

    def answer_query(self, query: str, top_k: int = 2) -> str:
        """Pipeline execution: retrieves context and generates a grounded support response."""
        retrieved_chunks = self.retrieve_context(query, top_k=top_k)
        
        context_texts = []
        for index, chunk in enumerate(retrieved_chunks, start=1):
            title = chunk["metadata"].get("title", "Unknown")
            content = chunk["content"]
            context_texts.append(f"Source [{index}] (Title: {title}):\n{content}")
        
        combined_context = "\n\n".join(context_texts)

        system_prompt = (
            "You are a technical support and operations assistant for the educational platform Northstar. "
            "Answer the user's question accurately using ONLY the provided context below. "
            "If the answer cannot be found in the context, politely state that you do not have enough information "
            "in the knowledge base to resolve the issue."
        )

        user_prompt = (
            f"Context Information:\n{combined_context}\n\n"
            f"User Question: {query}\n\n"
            "Provide a helpful, clear, and professional support response based strictly on the context above."
        )

        response = self.openai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
        )

        return response.choices[0].message.content


if __name__ == "__main__":
    print("==================================================")
    print(" Northstar Technical Support & Operations Assistant")
    print("==================================================")
    
    assistant = NorthstarRAGAssistant()
    print("\n[Ready] Assistant initialized successfully. Type 'exit' or 'quit' to close.\n")
    
    while True:
        try:
            user_query = input("Write here your question: ")
            if not user_query.strip():
                continue
            
            if user_query.strip().lower() in ["exit", "quit"]:
                print("\nExiting Northstar assistant. Goodbye!")
                break
            
            print("\n[Info] Processing query and generating response...")
            answer = assistant.answer_query(user_query, top_k=2)
            
            print("\n=== Generated Support Response ===")
            print(answer)
            print("-" * 50 + "\n")
            
        except KeyboardInterrupt:
            print("\nExiting Northstar assistant. Goodbye!")
            break
        except Exception as exc:
            print(f"\n[Error] An error occurred: {exc}\n")