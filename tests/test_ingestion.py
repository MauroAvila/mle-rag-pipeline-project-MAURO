"""
Unit Tests for Phase 1: Data Ingestion, JSONL Structure, and Persistent Report Validation
Validates that processed chunks and persistent metrics reports meet all specifications.
"""

import json
from pathlib import Path
import pytest


@pytest.fixture
def processed_chunks_file():
    """Fixture providing the path to the processed chunks JSONL file."""
    return Path("data/processed/processed_chunks.jsonl")


@pytest.fixture
def ingestion_report_file():
    """Fixture providing the path to the persistent ingestion report JSON file."""
    return Path("data/processed/ingestion_report.json")


def test_processed_chunks_exist(processed_chunks_file):
    """Test that the processed chunks output file exists on disk."""
    assert processed_chunks_file.exists(), f"Processed chunks file not found at {processed_chunks_file}"


def test_jsonl_structure_and_metadata(processed_chunks_file):
    """Test that every record in the JSONL file contains all required keys and valid data types."""
    if not processed_chunks_file.exists():
        pytest.skip("Processed chunks file does not exist yet. Run ingestion first.")

    required_keys = {"chunk_id", "document_id", "title", "chunk_index", "content"}
    
    line_count = 0
    with open(processed_chunks_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            
            line_count += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                pytest.fail(f"Line {line_num} is not valid JSON format: {e}")

            # Check that all required metadata keys are present
            missing_keys = required_keys - set(record.keys())
            assert not missing_keys, f"Line {line_num} is missing required metadata keys: {missing_keys}"

            # Validate field types and non-empty constraints
            assert isinstance(record["chunk_id"], str) and record["chunk_id"].strip(), f"Line {line_num}: invalid chunk_id"
            assert isinstance(record["document_id"], str) and record["document_id"].strip(), f"Line {line_num}: invalid document_id"
            assert isinstance(record["title"], str) and record["title"].strip(), f"Line {line_num}: invalid title"
            assert isinstance(record["chunk_index"], int) and record["chunk_index"] >= 0, f"Line {line_num}: invalid chunk_index"
            assert isinstance(record["content"], str) and record["content"].strip(), f"Line {line_num}: content field is empty"

    assert line_count > 0, "The processed chunks JSONL file is empty."


def test_ingestion_report_exists_and_structure(ingestion_report_file):
    """Test that the persistent ingestion report exists, is valid JSON, and contains correct metric attributes."""
    assert ingestion_report_file.exists(), f"Ingestion report file not found at {ingestion_report_file}"

    try:
        with open(ingestion_report_file, "r", encoding="utf-8") as f:
            report = json.load(f)
    except json.JSONDecodeError as e:
        pytest.fail(f"Ingestion report file is not valid JSON format: {e}")

    required_report_keys = {
        "timestamp",
        "total_documents_processed",
        "total_chunks_generated",
        "average_chunk_length_words",
        "minimum_chunk_length_words",
        "maximum_chunk_length_words",
        "metadata_completeness_percentage"
    }

    missing_keys = required_report_keys - set(report.keys())
    assert not missing_keys, f"Ingestion report is missing required metric keys: {missing_keys}"

    # Validate data types and logical ranges
    assert isinstance(report["timestamp"], str) and report["timestamp"].strip(), "Invalid report timestamp"
    assert isinstance(report["total_documents_processed"], int) and report["total_documents_processed"] >= 0, "Invalid total_documents_processed"
    assert isinstance(report["total_chunks_generated"], int) and report["total_chunks_generated"] >= 0, "Invalid total_chunks_generated"
    assert isinstance(report["average_chunk_length_words"], (int, float)) and report["average_chunk_length_words"] >= 0, "Invalid average_chunk_length_words"
    assert isinstance(report["minimum_chunk_length_words"], int) and report["minimum_chunk_length_words"] >= 0, "Invalid minimum_chunk_length_words"
    assert isinstance(report["maximum_chunk_length_words"], int) and report["maximum_chunk_length_words"] >= 0, "Invalid maximum_chunk_length_words"
    assert isinstance(report["metadata_completeness_percentage"], (int, float)) and 0 <= report["metadata_completeness_percentage"] <= 100.0, "Invalid metadata_completeness_percentage range"