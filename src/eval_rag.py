import json
import re
import sys
from typing import List


def normalize(text: str) -> str:
    """
    Normalize answers for fair comparison.
    Lowercase, remove punctuation and articles, collapse whitespace.
    """
    if text is None:
        return ""

    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = " ".join(text.split())
    return text


def extract_final_answer(model_answer: str) -> str:
    """
    Extract the model's final answer.
    Assumes chat-style output with 'ASSISTANT:'.
    """
    if model_answer is None:
        return ""

    if "ASSISTANT:" in model_answer:
        return model_answer.split("ASSISTANT:")[-1].strip()

    return model_answer.strip()


def exact_match(pred: str, golds: List[str]) -> bool:
    pred_norm = normalize(pred)
    for g in golds:
        if pred_norm == normalize(g):
            return True
    return False


def evaluate(results_path: str):
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    total = 0
    correct = 0
    skipped = 0

    for ex in results:
        # Skip failed generations
        if "error" in ex:
            skipped += 1
            continue

        model_answer = extract_final_answer(ex.get("model_answer", ""))
        gold_answers = [
            g["answer"] for g in ex.get("gold_answers", [])
            if "answer" in g
        ]

        if not gold_answers:
            skipped += 1
            continue

        total += 1
        if exact_match(model_answer, gold_answers):
            correct += 1

    accuracy = correct / total if total > 0 else 0.0

    print("==== RAG Evaluation ====")
    print(f"Total evaluated: {total}")
    print(f"Correct (EM):    {correct}")
    print(f"Skipped:         {skipped}")
    print(f"Exact Match:     {accuracy:.4f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python eval_rag.py <results.json>")
        sys.exit(1)

    evaluate(sys.argv[1])
