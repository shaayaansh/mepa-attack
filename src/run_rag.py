import os
import json
import gzip
from tqdm import tqdm
from PIL import Image, UnidentifiedImageError

from src.retriever import Retriever
from src.generator import Generator
from src.rag_model import RAGModel
from src.utils import load_mmqa_json, load_images_from_metadata

# =====================
# Configuration
# =====================

CACHE_DIR = "/scratch/shayan/hf_cache"

DATASET_NAME = "mmqa-mmpoisonrag"
DATASET_ROOT = f"datasets/{DATASET_NAME}"

IMAGE_METADATA_PATH = f"{DATASET_ROOT}/MMQA_image_metadata.json"
DATA_PATH = f"{DATASET_ROOT}/MMQA_test_image.json"

IMAGE_DIR = "datasets/mmqa/final_dataset_images" 

RETRIEVER_ID = "openai/clip-vit-base-patch32"
GENERATOR_ID = "llava-hf/llava-1.5-7b-hf"

OUTPUT_FILE = "results/rag_clip_llava_mmqa_clean_caption_baseline.json"


def main():

    print("Loading image metadata...")
    with open(IMAGE_METADATA_PATH, "r") as f:
        image_metadata = json.load(f)
    
    data = load_mmqa_json(DATA_PATH)
    print(f"Loaded {len(data)} MMQA test examples")

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

        images, image_ids = load_images_from_metadata(
            IMAGE_DIR,
            ex["metadata"]["image_doc_ids"],
            image_metadata
        )

        # Load candidate texts
        # Use image captions as the text modality
        texts = [
            image_metadata[img_id]["caption"]
            for img_id in image_ids
            if img_id in image_metadata and image_metadata[img_id].get("caption")
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

        retrieved_image_ids = [
            image_ids[i] for i in output["retrieved_image_indices"]
        ]

        retrieved_captions = [
            texts[i] for i in output["retrieved_text_indices"]
        ]

        results.append({
            "qid": ex.get("qid"),
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
        })

    print(f"Saving results to {OUTPUT_FILE}")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
