import os
import json
from tqdm import tqdm
from PIL import Image

from src.retriever import Retriever
from src.generator import Generator
from src.rag_model import RAGModel


# =========================
# Configuration
# =========================

CACHE_DIR = "/scratch/shayan/hf_cache"

DATASET_NAME = "webqa-mmpoisonrag"
DATASET_ROOT = f"datasets/{DATASET_NAME}"

DATA_PATH = f"{DATASET_ROOT}/WebQA_test_image.json"
IMAGE_DIR = f"{DATASET_ROOT}/extracted_images"

POISONED_METADATA_PATH = f"{DATASET_ROOT}/WebQA_image_metadata_poisoned.json"

RETRIEVER_ID = "openai/clip-vit-base-patch32"
GENERATOR_ID = "llava-hf/llava-1.5-7b-hf"

USE_POISONED_CAPTIONS = True  # False = baseline, True = attack

OUTPUT_FILE = (
    "results/rag_clip_llava_webqa_poisoned.json"
    if USE_POISONED_CAPTIONS
    else "results/rag_clip_llava_webqa_clean_caption_baseline.json"
)


def load_webqa_image(image_id):
    path = os.path.join(IMAGE_DIR, f"{image_id}.jpg")
    return Image.open(path).convert("RGB")


def main():

    poisoned_metadata = None
    if USE_POISONED_CAPTIONS:
        print("Loading poisoned metadata...")
        with open(POISONED_METADATA_PATH, "r") as f:
            poisoned_metadata = json.load(f)

    print("Loading WebQA data...")
    with open(DATA_PATH, "r") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} WebQA test examples")

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
    for guid, ex in tqdm(data.items()):

        question = ex["Q"].strip('"')
        gold_answers = [a.strip('"') for a in ex["A"]]

        evidence = ex.get("img_posFacts", []) + ex.get("img_negFacts", [])

        images = []
        texts = []
        image_ids = []

        for img in evidence:
            img_id = img["image_id"]
            caption = img["caption"]

            try:
                image = load_webqa_image(img_id)
            except Exception:
                continue

            images.append(image)
            texts.append(caption)
            image_ids.append(img_id)

        if not images or not texts:
            continue

        # Inject exactly ONE poisoned caption
        injected_poison = None

        if USE_POISONED_CAPTIONS and poisoned_metadata is not None:
            q_norm = " ".join(question.split()).strip().lower()

            for img_id in image_ids:
                key = str(img_id)   # type normalization

                if key not in poisoned_metadata:
                    continue

                for poison_entry in poisoned_metadata[key].get("poisoned", []):
                    pq = poison_entry.get("query", "")
                    pq_norm = " ".join(pq.split()).strip().lower()

                    # robust query matching
                    if pq_norm == q_norm:
                        injected_poison = poison_entry["poisoned_candidates"][0]
                        texts.append(injected_poison)
                        break

                if injected_poison is not None:
                    break


        try:
            output = rag.generate(
                question=question,
                images=images,
                texts=texts,
                max_new_tokens=150
            )
        except Exception as e:
            results.append({
                "qid": guid,
                "question": question,
                "error": str(e)
            })
            continue

        retrieved_image_ids = [
            image_ids[i] for i in output["retrieved_image_indices"]
        ]

        retrieved_captions = [
            texts[i] for i in output["retrieved_text_indices"]
        ]

        results.append({
            "qid": guid,
            "question": question,
            "model_answer": output["answer"],
            "gold_answers": gold_answers,

            # Candidate pool
            "associated_images": image_ids,
            "associated_captions": texts,

            "retrieved_image_ids": retrieved_image_ids,
            "retrieved_captions": retrieved_captions,
            "image_scores": output["image_scores"],
            "text_scores": output["text_scores"],

            # Attack bookkeeping
            "poison_injected": injected_poison is not None,
            "poison_caption": injected_poison,
        })

    print(f"Saving results to {OUTPUT_FILE}")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
