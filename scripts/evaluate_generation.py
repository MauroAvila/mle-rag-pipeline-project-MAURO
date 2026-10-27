"""
Phase 4 Evaluation: Generation & Groundedness Metrics (Faithfulness & Key Facts Coverage)
Evaluates LLM generation quality and hallucination control using Mistral as a judge.
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


def evaluate_generation(
    dataset_path: str = "data/evaluation/questions.json",
    db_path: str = "data/vector_store",
    report_path: str = "data/processed/generation_evaluation_report.json",
    model_name: str = "all-MiniLM-L6-v2",
    collection_name: str = "northstar_support"
) -> None:
    """Evaluates generation faithfulness and key facts coverage using an LLM-as-a-judge method."""
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

    supported_cases = 0
    faithfulness_scores = []
    key_facts_scores = []

    print(f"[Info] Evaluating generation metrics on {len(dataset)} items...")

    for item in dataset:
        query = item.get("query") or item.get("question")
        expected_facts = item.get("expected_key_facts", [])
        should_answer = item.get("should_answer", True)

        if not should_answer or not expected_facts:
            continue  # Skip abstention cases for generation-specific checks

        supported_cases += 1

        # Retrieve context
        q_emb = encoder.encode([query]).tolist()
        results = collection.query(query_embeddings=q_emb, n_results=2)
        docs = results["documents"][0] if results and "documents" in results else []
        combined_context = "\n".join(docs)

        # Generate response
        sys_prompt = "You are a support assistant. Answer accurately using ONLY the provided context."
        user_prompt = f"Context:\n{combined_context}\n\nQuestion: {query}"
        
        try:
            resp = openai_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
                temperature=0.0
            )
            generated_answer = resp.choices[0].message.content
        except Exception:
            generated_answer = ""

        # LLM-as-a-Judge evaluation prompt for Faithfulness & Key Facts
        judge_prompt = (
            f"You are an objective AI evaluation judge.\n"
            f"Context Provided:\n{combined_context}\n\n"
            f"Generated Answer:\n{generated_answer}\n\n"
            f"Expected Key Facts to cover: {expected_facts}\n\n"
            f"Evaluate two metrics and output STRICT JSON format with keys:\n"
            f"1. 'faithful' (boolean: true if all claims are supported by context without hallucination, false otherwise)\n"
            f"2. 'facts_covered_ratio' (float between 0.0 and 1.0 representing proportion of expected key facts addressed)."
        )

        try:
            judge_resp = openai_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=0.0
            )
            content = judge_resp.choices[0].message.content.strip()
            # Clean markdown code blocks if present
            if content.startswith("```json"):
                content = content[7:-3].strip()
            elif content.startswith("```"):
                content = content[3:-3].strip()
            
            evaluation = json.loads(content)
            faithfulness_scores.append(1.0 if evaluation.get("faithful", False) else 0.0)
            key_facts_scores.append(float(evaluation.get("facts_covered_ratio", 0.0)))
        except Exception:
            faithfulness_scores.append(0.0)
            key_facts_scores.append(0.0)

    avg_faithfulness = (sum(faithfulness_scores) / supported_cases * 100) if supported_cases > 0 else 0.0
    avg_facts_coverage = (sum(key_facts_scores) / supported_cases * 100) if supported_cases > 0 else 0.0

    report_data = {
        "dataset_used": str(dataset_file),
        "supported_queries_evaluated": supported_cases,
        "faithfulness_percentage": round(avg_faithfulness, 2),
        "key_facts_coverage_percentage": round(avg_facts_coverage, 2)
    }

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=4, ensure_ascii=False)

    print(f"[Success] Generation evaluation report saved to {report_path}")

    # Print Console Summary
    print("\n==============================================")
    print("         GENERATION EVALUATION REPORT         ")
    print("==============================================")
    print(f"• Supported Queries Evaluated: {supported_cases}")
    print(f"• Faithfulness / Groundedness: {avg_faithfulness:.1f}%")
    print(f"• Key Facts Coverage Ratio   : {avg_facts_coverage:.1f}%")
    print("==============================================\n")


if __name__ == "__main__":
    print("--- Starting Phase 4 Evaluation: Generation & Groundedness ---")
    evaluate_generation()