"""
Phase 1: Data Ingestion, Cleaning, Chunking, Metrics Validation, and Persistent Reporting
Module to process raw Markdown knowledge base files for the Northstar RAG pipeline.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any


def clean_text(text: str) -> str:
    """Cleans raw markdown text by normalizing whitespace and removing unwanted artifacts."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Splits text into chunks of specified word count with a sliding overlap."""
    words = text.split()
    if not words:
        return []
    
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        if end >= len(words):
            break
        start += chunk_size - overlap
    return chunks


def process_knowledge_base(
    input_dir: str = "data/knowledge_base",
    output_path: str = "data/processed/processed_chunks.jsonl",
    report_path: str = "data/processed/ingestion_report.json",
    chunk_size: int = 500,
    overlap: int = 50
) -> List[Dict[str, Any]]:
    """Scans markdown files, cleans them, chunks them, validates metadata, and saves to JSONL and report."""
    input_path = Path(input_dir)
    output_file = Path(output_path)
    report_file = Path(report_path)
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"[Warning] Input directory '{input_dir}' does not exist. Creating it as a placeholder.")
        input_path.mkdir(parents=True, exist_ok=True)

    all_chunks = []
    markdown_files = list(input_path.glob("*.md"))

    print(f"[Info] Found {len(markdown_files)} Markdown document(s) to process.")

    for file_path in markdown_files:
        document_id = file_path.stem
        raw_content = file_path.read_text(encoding="utf-8")
        cleaned_content = clean_text(raw_content)

        # Extract title from first header line if available
        title = document_id.replace("-", " ").title()
        lines = cleaned_content.splitlines()
        if lines and lines[0].startswith("#"):
            title = lines[0].lstrip("#").strip()

        text_chunks = chunk_text(cleaned_content, chunk_size=chunk_size, overlap=overlap)

        for idx, chunk_content in enumerate(text_chunks):
            chunk_id = f"{document_id}_chunk_{idx}"
            
            # Validation check: Ensure chunk content is not empty
            assert chunk_content.strip(), f"Error: Generated empty chunk for {chunk_id}"

            chunk_record = {
                "chunk_id": chunk_id,
                "document_id": document_id,
                "title": title,
                "chunk_index": idx,
                "content": chunk_content
            }
            all_chunks.append(chunk_record)

    # Save processed chunks to JSONL
    with open(output_file, "w", encoding="utf-8") as f:
        for record in all_chunks:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[Success] Processed chunks saved successfully to {output_path}")

    # --- Calculate Ingestion Metrics ---
    total_chunks = len(all_chunks)
    metrics_data = {
        "timestamp": datetime.now().isoformat(),
        "total_documents_processed": len(markdown_files),
        "total_chunks_generated": total_chunks,
        "average_chunk_length_words": 0.0,
        "minimum_chunk_length_words": 0,
        "maximum_chunk_length_words": 0,
        "metadata_completeness_percentage": 100.0 if total_chunks > 0 else 0.0
    }

    if total_chunks > 0:
        chunk_lengths = [len(c["content"].split()) for c in all_chunks]
        metrics_data["average_chunk_length_words"] = round(sum(chunk_lengths) / total_chunks, 1)
        metrics_data["minimum_chunk_length_words"] = min(chunk_lengths)
        metrics_data["maximum_chunk_length_words"] = max(chunk_lengths)

        required_keys = {"chunk_id", "document_id", "title", "chunk_index", "content"}
        complete_records = sum(1 for c in all_chunks if required_keys.issubset(c.keys()))
        metrics_data["metadata_completeness_percentage"] = round((complete_records / total_chunks) * 100, 1)

    # Save metrics report persistently to JSON file
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=4, ensure_ascii=False)

    print(f"[Success] Persistent ingestion report saved successfully to {report_path}")

    # --- Print Console Summary Report ---
    print("\n==============================================")
    print("          INGESTION METRICS REPORT            ")
    print("==============================================")
    print(f"• Total Chunks Generated    : {metrics_data['total_chunks_generated']}")
    if total_chunks > 0:
        print(f"• Average Chunk Length      : {metrics_data['average_chunk_length_words']} words")
        print(f"• Minimum Chunk Length      : {metrics_data['minimum_chunk_length_words']} words")
        print(f"• Maximum Chunk Length      : {metrics_data['maximum_chunk_length_words']} words")
        print(f"• Metadata Completeness     : {metrics_data['metadata_completeness_percentage']}%")
    else:
        print("• [Warning] No chunks were generated. Please verify source files.")
    print("==============================================\n")

    return all_chunks


if __name__ == "__main__":
    print("--- Starting Phase 1: Data Ingestion & Validation (Northstar) ---")
    process_knowledge_base()