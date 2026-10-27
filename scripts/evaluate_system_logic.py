"""
Phase 5 Evaluation: System Logic & Control Metrics (Decision Accuracy & Fallback Rate)
Evaluates whether the system correctly triggers abstention or responds based on domain support.
"""

import json
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from sentence_transformers import SentenceTransformer

load_dotenv()
MODEL_NAME = "ministral-14b-latest"
MISTRAL_BASE_URL = "https://api.mistral.ai/v1"


def evaluate_system_logic(
    dataset_path: str = "data/evaluation/questions.json",
    db_path: str = "data/vector_store",
    report_path: str = "data/processed/system_evaluation_report.json",
    model_name: str = "all-MiniLM-L6-v2",
    collection_name: str = "northstar_support"
) -> None:
    """Evaluates decision accuracy and fallback/abstention rate."""
    dataset_file = Path(dataset_path)
    db_dir = Path(db_path)
    report_file = Path(report_path)
    
    report_file.parent.mkdir(parents=True, exist_ok=True)
    api_key = os.getenv("MISTRAL_API_KEY")

    if not dataset_file.exists() or not api_key:
        print("[Error] Evaluation dataset missing or MISTRAL_API_KEY not configured.")
        return

    with open(dataset_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    openai_client = OpenAI(api_key=api_key, base_url=MISTRAL_BASE_URL)
    encoder = SentenceTransformer(model_name)
    client = chromadb.PersistentClient(path=str(db_dir))
    collection = client.get_collection(name=collection_name)

    total_cases = len(dataset)
    correct_decisions = 0
    abstention_triggered_count = 0
    expected_abstention_count = 0

    print(f"[Info] Evaluating system logic and decision accuracy on {total_cases} items...")

    for item in dataset:
        query = item.get("query") or item.get("question")
        should_answer = item.get("should_answer", True)

        if not should_answer:
            expected_abstention_count += 1

        # Retrieve context
        q_emb = encoder.encode([query]).tolist()
        results = collection.query(query_embeddings=q_emb, n_results=2)
        docs = results["documents"][0] if results and "documents" in results else []
        combined_context = "\n".join(docs)

        sys_prompt = (
            "You are a support assistant for Northstar. Answer using ONLY the provided context. "
            "If the answer cannot be found in the context, politely state that you do not have enough information."
        )
        user_prompt = f"Context:\n{combined_context}\n\nQuestion: {query}"

        try:
            resp = openai_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
                temperature=0.0
            )
            answer = resp.choices[0].message.content.lower()
        except Exception:
            answer = ""

        # Check if model abstained (triggered fallback phrase)
        fallback_phrases = ["do not have enough information", "not enough information", "cannot find"]
        did_abstain = any(phrase in answer for phrase in fallback_phrases)

        if did_abstain:
            abstention_triggered_count += 1

        # Decision is correct if system answered when it should, or abstained when it should not
        if (should_answer and not did_abstain) or (not should_answer and did_abstain):
            correct_decisions += 1

    decision_accuracy = (correct_decisions / total_cases) * 100 if total_cases > 0 else 0.0

    report_data = {
        "dataset_used": str(dataset_file),
        "total_cases_evaluated": total_cases,
        "decision_accuracy_percentage": round(decision_accuracy, 2),
        "expected_abstentions": expected_abstention_count,
        "actual_abstentions_triggered": abstention_triggered_count
    }

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=4, ensure_ascii=False)

    print(f"[Success] System evaluation report saved to {report_path}")

    # Print Console Summary
    print("\n==============================================")
    print("          SYSTEM LOGIC EVALUATION REPORT      ")
    print("==============================================")
    print(f"• Total Cases Evaluated     : {total_cases}")
    print(f"• Decision Accuracy         : {decision_accuracy:.1f}%")
    print(f"• Expected Abstentions      : {expected_abstention_count}")
    print(f"• Actual Abstentions Triggered: {abstention_triggered_count}")
    print("==============================================\n")


if __name__ == "__main__":
    print("--- Starting Phase 5 Evaluation: System Logic & Control ---")
    evaluate_system_logic()