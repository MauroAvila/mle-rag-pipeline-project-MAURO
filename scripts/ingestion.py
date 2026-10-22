"""
Phase 1: Ingestion, Cleaning, and Chunking
Module to load, clean, chunk Markdown files, and persist the output to disk in JSONL format.
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List


class KnowledgeBaseIngestor:
    """Handles loading, cleaning, chunking Markdown files, and saving processed chunks to JSONL."""

    def __init__(
        self,
        kb_path: str = "data/knowledge_base",
        output_path: str = "data/processed_chunks.jsonl",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> None:
        self.kb_path = Path(kb_path)
        self.output_path = Path(output_path)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_documents(self) -> List[Dict[str, Any]]:
        """Scans the knowledge base directory and loads all Markdown documents."""
        documents = []
        if not self.kb_path.exists():
            print(f"[Warning] Knowledge base directory '{self.kb_path}' does not exist.")
            return documents

        for file_path in self.kb_path.glob("*.md"):
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    content = file.read()
                    doc_id = file_path.stem
                    documents.append({
                        "document_id": doc_id,
                        "title": doc_id.replace("_", " ").title(),
                        "file_path": str(file_path),
                        "content": content,
                    })
            except Exception as exc:
                print(f"[Error] Failed to read {file_path}: {exc}")

        return documents

    def clean_text(self, text: str) -> str:
        """Normalizes spaces and removes excessive consecutive line breaks."""
        text = re.sub(r"\n+", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    def chunk_text(self, text: str) -> List[str]:
        """Splits text into chunks using word sliding window with specified overlap."""
        words = text.split()
        if not words:
            return []

        chunks = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        for i in range(0, len(words), step):
            chunk_words = words[i : i + self.chunk_size]
            chunks.append(" ".join(chunk_words))
        return chunks

    def process(self) -> List[Dict[str, Any]]:
        """Runs the complete ingestion pipeline: load, clean, chunk, and annotate."""
        raw_documents = self.load_documents()
        processed_chunks = []

        for doc in raw_documents:
            cleaned_content = self.clean_text(doc["content"])
            text_chunks = self.chunk_text(cleaned_content)

            for index, chunk in enumerate(text_chunks):
                processed_chunks.append({
                    "chunk_id": f"{doc['document_id']}_chunk_{index}",
                    "document_id": doc["document_id"],
                    "title": doc["title"],
                    "content": chunk,
                    "chunk_index": index,
                })

        return processed_chunks

    def save_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """Saves the processed chunks into a JSONL file."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as file:
            for chunk in chunks:
                file.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        print(f"[Success] Processed chunks successfully saved to '{self.output_path}'")


if __name__ == "__main__":
    print("--- Starting Phase 1: Ingestion, Cleaning & Chunking (Northstar) ---")
    ingestor = KnowledgeBaseIngestor()
    chunks = ingestor.process()

    print(f"\n[Result] Total chunks created: {len(chunks)}")
    if chunks:
        ingestor.save_chunks(chunks)