"""
Phase 3 Evaluation: Retrieval Metrics (HitRate@k and MRR)
Evaluates the local ChromaDB retrieval performance using a calibration or evaluation dataset.
"""

import json
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer


def evaluate_retriever(
    dataset_path: str = "data/evaluation/questions.json",
    db_path: str = "data/vector_store",
    report_path: str = "data/processed/retrieval_evaluation_report.json",
    model_name: str = "all-MiniLM-L6-v2",
    collection_name: str = "northstar_support",
    top_k: int = 3
) -> None:
    """Evaluates HitRate@k and MRR for retrieval accuracy."""
    dataset_file = Path(dataset_path)
    db_dir = Path(db_path)
    report_file = Path(report_path)
    
    report_file.parent.mkdir(parents=True, exist_ok=True)

    if not dataset_file.exists():
        print(f"[Error] Evaluation dataset not found at {dataset_path}. Please provide calibration.json or questions.json.")
        return

    if not db_dir.exists():
        print(f"[Error] Vector database not found at {db_path}. Run Phase 2 vectorization first.")
        return

    # Load dataset
    with open(dataset_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # Initialize encoder and DB client
    encoder = SentenceTransformer(model_name)
    client = chromadb.PersistentClient(path=str(db_dir))
    collection = client.get_collection(name=collection_name)

    total_queries = len(dataset)
    hits_at_k = 0
    reciprocal_ranks = []

    print(f"[Info] Evaluating retriever on {total_queries} query/queries (Top-{top_k})...")

    for item in dataset:
        query = item.get("query") or item.get("question")
        # Support both 'expected_chunk_ids' or 'expected_documents' keys in JSON schema
        expected_chunks = item.get("expected_chunk_ids") or item.get("expected_documents", [])
        
        if not query:
            continue

        # Execute vector query
        query_embedding = encoder.encode([query]).tolist()
        results = collection.query(query_embeddings=query_embedding, n_results=top_k)
        
        retrieved_ids = results["ids"][0] if results and "ids" in results else []

        # Check HitRate@k (if any expected chunk is present in retrieved results)
        hit_found = False
        first_match_rank = None

        for rank, chunk_id in enumerate(retrieved_ids, start=1):
            if chunk_id in expected_chunks:
                hit_found = True
                if first_match_rank is None:
                    first_match_rank = rank
                break

        if hit_found:
            hits_at_k += 1

        # Calculate Reciprocal Rank for MRR
        if first_match_rank is not None:
            reciprocal_ranks.append(1.0 / first_match_rank)
        else:
            reciprocal_ranks.append(0.0)

    # Compute final metrics
    hit_rate_at_k = (hits_at_k / total_queries) * 100 if total_queries > 0 else 0.0
    mrr = (sum(reciprocal_ranks) / total_queries) if total_queries > 0 else 0.0

    report_data = {
        "dataset_used": str(dataset_file),
        "total_queries_evaluated": total_queries,
        "top_k": top_k,
        "hit_rate_percentage": round(hit_rate_at_k, 2),
        "mean_reciprocal_rank_mrr": round(mrr, 4)
    }

    # Save report persistently
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=4, ensure_ascii=False)

    print(f"[Success] Retrieval evaluation report saved to {report_path}")

    # Print Console Summary
    print("\n==============================================")
    print("         RETRIEVAL EVALUATION REPORT          ")
    print("==============================================")
    print(f"• Dataset Evaluated         : {dataset_file.name}")
    print(f"• Total Queries             : {total_queries}")
    print(f"• HitRate@{top_k}               : {hit_rate_at_k}%")
    print(f"• MRR (Mean Reciprocal Rank): {mrr:.4f}")
    print("==============================================\n")


if __name__ == "__main__":
    print("--- Starting Phase 3 Evaluation: Retriever Performance ---")
    evaluate_retriever()