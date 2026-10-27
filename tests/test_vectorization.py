"""
Unit Tests for Phase 2: Vectorization, ChromaDB Indexing, and Persistent Report Validation
Validates vector store creation, collection synchronization, and persistent vectorization reports.
"""

import json
from pathlib import Path
import chromadb
import pytest


@pytest.fixture
def vector_db_dir():
    """Fixture providing the path to the ChromaDB persistent directory."""
    return Path("data/vector_store")


@pytest.fixture
def vectorization_report_file():
    """Fixture providing the path to the persistent vectorization report JSON file."""
    return Path("data/processed/vectorization_report.json")


def test_vector_db_exists(vector_db_dir):
    """Test that the vector database directory exists on disk."""
    assert vector_db_dir.exists(), f"Vector database directory not found at {vector_db_dir}"


def test_vector_collection_and_count(vector_db_dir):
    """Test that the ChromaDB collection exists and contains indexed items."""
    if not vector_db_dir.exists():
        pytest.skip("Vector database directory does not exist yet. Run vectorization first.")

    client = chromadb.PersistentClient(path=str(vector_db_dir))
    collections = client.list_collections()
    collection_names = [col.name for col in collections]

    assert "northstar_support" in collection_names, "Collection 'northstar_support' not found in vector store."

    collection = client.get_collection(name="northstar_support")
    count = collection.count()
    assert count > 0, "ChromaDB collection is empty."


def test_vectorization_report_exists_and_structure(vectorization_report_file):
    """Test that the persistent vectorization report exists, is valid JSON, and contains correct metrics."""
    assert vectorization_report_file.exists(), f"Vectorization report file not found at {vectorization_report_file}"

    try:
        with open(vectorization_report_file, "r", encoding="utf-8") as f:
            report = json.load(f)
    except json.JSONDecodeError as e:
        pytest.fail(f"Vectorization report file is not valid JSON format: {e}")

    required_report_keys = {
        "timestamp",
        "model_name",
        "embedding_dimension",
        "collection_name",
        "total_chunks_loaded",
        "total_chunks_indexed",
        "synchronization_status"
    }

    missing_keys = required_report_keys - set(report.keys())
    assert not missing_keys, f"Vectorization report is missing required metric keys: {missing_keys}"

    # Validate data types and logical ranges
    assert isinstance(report["timestamp"], str) and report["timestamp"].strip(), "Invalid report timestamp"
    assert isinstance(report["model_name"], str) and report["model_name"].strip(), "Invalid model name"
    assert isinstance(report["embedding_dimension"], int) and report["embedding_dimension"] > 0, "Invalid embedding dimension"
    assert isinstance(report["collection_name"], str) and report["collection_name"].strip(), "Invalid collection name"
    assert isinstance(report["total_chunks_loaded"], int) and report["total_chunks_loaded"] >= 0, "Invalid total_chunks_loaded"
    assert isinstance(report["total_chunks_indexed"], int) and report["total_chunks_indexed"] >= 0, "Invalid total_chunks_indexed"
    assert isinstance(report["synchronization_status"], bool), "Invalid synchronization_status type"
    assert report["synchronization_status"] is True, "Vector store count does not match loaded chunks count."