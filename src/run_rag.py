import os
import json
import gzip
from tqdm import tqdm
from PIL import Image, UnidentifiedImageError

from retriever import Retriever
from generator import Generator
from rag_model import RAGModel


# =====================
# Configuration
# =====================

CACHE_DIR = "/scratch/shayan/hf_cache"
IMAGE_DIR = "final_dataset_images"

RETRIEVER_ID = "openai/clip-vit-base-patch32"
GENERATOR_ID = "llava-hf/llava-1.5-7b-hf"

FILTERED_DATA_PATH = "dataset/MMQA_train_image_text_only.jsonl.gz"
TEXTS_PATH = "dataset/MMQA_texts.jsonl.gz"

OUTPUT_FILE = f"rag_clip_llava_results.json"


# =====================
# Utility functions
# =====================

def load_filtered_data(path):
    data = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def load_text_corpus(path):
    corpus = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            corpus[ex["id"]] = ex["text"]
    return corpus


def load_images(image_ids):
    images = []
    valid_ids = []

    for img_id in image_ids:
        for ext in [".jpg", ".png", ".jpeg"]:
            p = os.path.join(IMAGE_DIR, img_id + ext)
            if not os.path.exists(p):
                continue

            try:
                img = Image.open(p)
                img = img.convert("RGB")
            except (UnidentifiedImageError, OSError) as e:
                # Corrupt or unreadable image → skip
                print(f"[WARN] Skipping image {img_id}{ext}: {e}")
                break

            images.append(img)
            valid_ids.append(img_id)
            break

    return images, valid_ids



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
        images, image_ids = load_images(ex["metadata"]["image_doc_ids"])

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
