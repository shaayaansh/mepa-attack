import json
import re
import sys
import argparse
from typing import List


BASE_RESULTS_DIR = "/scratch/shayan/Projects/mepa-attack/results"

RESULTS_TEMPLATE = {
    "mmqa": {
        "clean": "rag_clip_llava_mmqa_clean_caption_baseline.json",
        "poisoned": "rag_clip_llava_mmqa_poisoned.json",
    },
    "webqa": {
        "clean": "rag_clip_llava_webqa_clean_caption_baseline.json",
        "poisoned": "rag_clip_llava_webqa_poisoned.json",
    },
}

def normalize(text: str) -> str:
    """
    Normalize answers for fair comparison.
    Lowercase, remove punctuation and articles, collapse whitespace.
    """
    if not text:
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
    if not model_answer:
        return ""

    if "ASSISTANT:" in model_answer:
        return model_answer.split("ASSISTANT:")[-1].strip()

    return model_answer.strip()



def exact_match_mmqa(pred: str, golds: List[str]) -> bool:
    """
    MMQA: strict EM with light plural handling.
    """
    pred_norm = normalize(pred)

    for g in golds:
        gold_norm = normalize(g)

        if pred_norm == gold_norm:
            return True

        # Allow singular/plural mismatch
        if pred_norm.rstrip("s") == gold_norm.rstrip("s"):
            return True

    return False


def exact_match_webqa(pred: str, golds: List[str], qcate: str = None) -> bool:
    """
    WebQA: containment-based EM + Yes/No handling.
    """
    pred_norm = normalize(pred)

    for g in golds:
        gold_norm = normalize(g)

        # Yes/No questions
        if qcate == "YesNo":
            if pred_norm.startswith("yes") and gold_norm.startswith("yes"):
                return True
            if pred_norm.startswith("no") and gold_norm.startswith("no"):
                return True

        # Containment-based EM
        if pred_norm == gold_norm:
            return True
        if pred_norm in gold_norm:
            return True
        if gold_norm in pred_norm:
            return True

    return False



def evaluate(dataset: str, split: str):
    try:
        filename = RESULTS_TEMPLATE[dataset][split]
    except KeyError:
        raise ValueError(
            f"Invalid combination: dataset_name={dataset}, split={split}"
        )

    results_path = f"{BASE_RESULTS_DIR}/{filename}"

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

        gold_answers = ex.get("gold_answers", [])

        # MMQA golds are dicts; WebQA golds are strings
        if dataset == "mmqa":
            gold_answers = [
                g["answer"]
                for g in gold_answers
                if isinstance(g, dict) and "answer" in g
            ]

        if not model_answer or not gold_answers:
            skipped += 1
            continue

        total += 1

        if dataset == "mmqa":
            is_correct = exact_match_mmqa(model_answer, gold_answers)

        elif dataset == "webqa":
            qcate = ex.get("Qcate")  # may be None, that's fine
            is_correct = exact_match_webqa(
                model_answer, gold_answers, qcate=qcate
            )

        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        if is_correct:
            correct += 1

    accuracy = correct / total if total > 0 else 0.0

    print("==== RAG Evaluation ====")
    print(f"Dataset:         {dataset}")
    print(f"Results file:    {results_path}")
    print(f"Total evaluated: {total}")
    print(f"Correct:         {correct}")
    print(f"Skipped:         {skipped}")
    print(f"Accuracy:        {accuracy:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_name",
        choices=["mmqa", "webqa"],
        required=True,
        help="Dataset to evaluate"
    )

    parser.add_argument(
        "--split",
        choices=["clean", "poisoned"],
        required=True,
        help="Evaluate clean or poisoned results"
    )

    args = parser.parse_args()

    evaluate(args.dataset_name, args.split)

