import json
import os
from tqdm import tqdm
from openai import OpenAI

# =====================
# Configuration
# =====================

OPENAI_MODEL = "gpt-4.1-mini"

INPUT_FILES = {
    "mmqa": "/scratch/shayan/Projects/mepa-attack/results/clean_results/rag_clip_blip2_mmqa_clean_k=1.json",
    "webqa": "/scratch/shayan/Projects/mepa-attack/results/clean_results/rag_clip_blip2_webqa_clean_k=2.json",
}

OUTPUT_DIR = "/scratch/shayan/Projects/mepa-attack/results/paraphrased_questions"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =====================
# OpenAI Setup
# =====================

def load_openai_key(path="/scratch/shayan/Projects/mepa-attack/OpenAI_key.txt"):
    with open(path, "r") as f:
        return f.read().strip()

os.environ["OPENAI_API_KEY"] = load_openai_key()
client = OpenAI()


# =====================
# Prompt Builder
# =====================

def build_paraphrase_prompt(question: str) -> str:
    return f"""
    You are paraphrasing a user question for robustness evaluation of a retrieval-augmented generation (RAG) system.

    Original Question:
    "{question}"

    Instructions:
    - Rewrite the question using different wording.
    - Preserve the exact meaning.
    - The answer must remain identical.
    - Do NOT add or remove constraints.
    - Do NOT change specificity.
    - Return ONLY the paraphrased question.
    """.strip()


# =====================
# Paraphrase Function
# =====================

def paraphrase_question(question: str) -> str:
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": build_paraphrase_prompt(question)}],
        temperature=0.3  # low temperature for semantic stability
    )
    return response.choices[0].message.content.strip()


# =====================
# Main Processing
# =====================

def process_dataset(name: str, input_path: str):
    print(f"\nProcessing {name.upper()} dataset...")

    with open(input_path, "r") as f:
        data = json.load(f)

    paraphrase_entries = []

    for entry in tqdm(data):
        qid = entry["qid"]
        original_question = entry["question"]

        try:
            paraphrased = paraphrase_question(original_question)
        except Exception as e:
            print(f"Error for qid={qid}: {e}")
            paraphrased = None

        paraphrase_entries.append({
            "qid": qid,
            "original_question": original_question,
            "paraphrased_question": paraphrased
        })

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{name}_paraphrased_questions.json"
    )

    with open(output_path, "w") as f:
        json.dump(paraphrase_entries, f, indent=2)

    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    for dataset_name, file_path in INPUT_FILES.items():
        process_dataset(dataset_name, file_path)
