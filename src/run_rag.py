import os
import json
import gzip
from tqdm import tqdm
from PIL import Image, UnidentifiedImageError

from src.retriever import Retriever
from src.generator import Generator
from src.rag_model import RAGModel
from src.utils import load_filtered_data, load_images, load_text_corpus


# =====================
# Configuration
# =====================

CACHE_DIR = "/scratch/shayan/hf_cache"

DATASET_NAME = "mmqa"
DATASET_ROOT = f"datasets/{DATASET_NAME}"

RETRIEVER_ID = "openai/clip-vit-base-patch32"
GENERATOR_ID = "llava-hf/llava-1.5-7b-hf"

OUTPUT_FILE = f"results/rag_clip_llava_{DATASET_NAME}_results.json"

FILTERED_DATA_PATH = (
    f"{DATASET_ROOT}/MMQA_train_image_text_only.jsonl.gz"
)
TEXTS_PATH = (
    f"{DATASET_ROOT}/MMQA_texts.jsonl.gz"
)
IMAGE_DIR = (
    f"{DATASET_ROOT}/final_dataset_images"
)



def main():

    print("Loading data...")
    data = load_filtered_data(FILTERED_DATA_PATH)
    text_corpus = load_text_corpus(TEXTS_PATH)

    print(f"Loaded {len(data)} filtered examples")

    print("Initializing models...")
    retriever = Retriever(
        model_id=RETRIEVER_ID,
        cache_dir=CACHE_DIR
    )

    generator = Generator(
        model_id=GENERATOR_ID,
        cache_dir=CACHE_DIR
    )

    rag = RAGModel(
        retriever=retriever,
        generator=generator,
        top_k_images=3,
        top_k_texts=3
    )

    results = []

    print("Running RAG...")
    for ex in tqdm(data):

        question = ex["question"]
        gold_answers = ex.get("answers", [])

        # Load candidate images
        images, image_ids = load_images(IMAGE_DIR, ex["metadata"]["image_doc_ids"])

        # Load candidate texts
        texts = [
            text_corpus[doc_id]
            for doc_id in ex["metadata"]["text_doc_ids"]
            if doc_id in text_corpus
        ]

        if len(images) == 0 or len(texts) == 0:
            continue

        try:
            output = rag.generate(
                question=question,
                images=images,
                texts=texts,
                max_new_tokens=150
            )
        except Exception as e:
            # Fail gracefully, but log
            results.append({
                "qid": ex.get("qid"),
                "question": question,
                "error": str(e)
            })
            continue

        results.append({
            "qid": ex.get("qid"),
            "question": question,
            "model_answer": output["answer"],
            "gold_answers": gold_answers,

            # Original candidates
            "associated_images": image_ids,
            "associated_docs": ex["metadata"]["text_doc_ids"],

            # Retrieved evidence
            "retrieved_images": image_ids[:len(output["retrieved_images"])],
            "retrieved_docs": output["retrieved_texts"],
        })

    print(f"Saving results to {OUTPUT_FILE}")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
