import os

os.environ["DISABLE_TRANSFORMERS_STREAMING"] = "1"
os.environ["HF_HOME"] = "/scratch/shayan/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "/scratch/shayan/hf_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/scratch/shayan/hf_cache"

import argparse
import json
from tqdm import tqdm
from src.retriever import Retriever
from src.generator import Generator
from src.rag_model import RAGModel
from src.utils import load_mmqa_json, load_images_from_metadata

# Paths / config

CACHE_DIR = "/scratch/shayan/hf_cache"

DATASET_NAME = "mmqa-mmpoisonrag"
DATASET_ROOT = f"datasets/{DATASET_NAME}"

IMAGE_METADATA_PATH = f"{DATASET_ROOT}/MMQA_image_metadata.json"
POISONED_METADATA_PATH = f"{DATASET_ROOT}/MMQA_image_metadata_poisoned.json"
DATA_PATH = f"{DATASET_ROOT}/MMQA_test_image.json"
PARAPHRASE_PATH = (
    "/scratch/shayan/Projects/mepa-attack/results/paraphrased_questions/"
    "mmqa_paraphrased_questions.json"
)


IMAGE_DIR = "datasets/mmqa/final_dataset_images"

CLEAN_RESULTS_DIR = "/scratch/shayan/Projects/mepa-attack/results/clean_results"
POISON_RESULTS_DIR = "/scratch/shayan/Projects/mepa-attack/results/mmqa_results"
ROBUSTNESS_RESULTS_DIR = "/scratch/shayan/Projects/mepa-attack/results/robustness_results"

os.makedirs(CLEAN_RESULTS_DIR, exist_ok=True)
os.makedirs(POISON_RESULTS_DIR, exist_ok=True)
os.makedirs(ROBUSTNESS_RESULTS_DIR, exist_ok=True)


# Model grids
RETRIEVERS = {
    "clip": "openai/clip-vit-base-patch32",
    "openclip": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    "sigclip": "google/siglip-base-patch16-224",
    "flava": "facebook/flava-full"
}

GENERATORS = {
    "llava": "llava-hf/llava-1.5-7b-hf",
    # "blip2": "Salesforce/blip2-flan-t5-xl",
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_poison", action="store_true",
                        help="Use poisoned captions")
    parser.add_argument("--robustness", action="store_true",
                        help="Use paraphrased queries")
    parser.add_argument("--k", type=int, default=1)

    args = parser.parse_args()

    USE_POISONED_CAPTIONS = args.use_poison
    ROBUSTNESS_MODE = args.robustness
    K = args.k

    print("Loading image metadata...")
    with open(IMAGE_METADATA_PATH, "r") as f:
        image_metadata = json.load(f)

    poisoned_metadata = None
    if USE_POISONED_CAPTIONS:
        print("Loading poisoned image metadata...")
        with open(POISONED_METADATA_PATH, "r") as f:
            poisoned_metadata = json.load(f)

    print("Loading MMQA test data...")
    data = load_mmqa_json(DATA_PATH)
    print(f"Loaded {len(data)} examples")

    paraphrase_map = None
    if ROBUSTNESS_MODE:
        print("Loading paraphrased questions...")
        with open(PARAPHRASE_PATH, "r") as f:
            paraphrase_data = json.load(f)

        paraphrase_map = {
            entry["qid"]: entry["paraphrased_question"]
            for entry in paraphrase_data
        }

    
    # Loop over model configs
    for retriever_type, retriever_id in RETRIEVERS.items():
        for generator_type, generator_id in GENERATORS.items():

            print("\n" + "=" * 60)
            print(f"Running RAG with Retriever={retriever_type}, Generator={generator_type}")
            print("=" * 60)

            retriever = Retriever(
                model_type=retriever_type,
                model_id=retriever_id,
                cache_dir=CACHE_DIR
            )

            generator = Generator(
                model_type=generator_type,
                model_id=generator_id,
                cache_dir=CACHE_DIR
            )

            rag = RAGModel(
                retriever=retriever,
                generator=generator,
                top_k_images=K,
                top_k_texts=K
            )

            if ROBUSTNESS_MODE:
                save_dir = ROBUSTNESS_RESULTS_DIR
                suffix = "robustness"
            elif USE_POISONED_CAPTIONS:
                save_dir = POISON_RESULTS_DIR
                suffix = "poisoned"
            else:
                save_dir = CLEAN_RESULTS_DIR
                suffix = "clean"

            output_file = (
                f"{save_dir}/rag_{retriever_type}_{generator_type}_mmqa_{suffix}_k={K}.json"
            )

            results = []
            print("Running RAG inference...")
            for ex in tqdm(data):

                qid = ex.get("qid")
                if ROBUSTNESS_MODE:
                    question = paraphrase_map.get(qid, ex["question"])
                else:
                    question = ex["question"]

                gold_answers = ex.get("answers", [])

                images, image_ids = load_images_from_metadata(
                    IMAGE_DIR,
                    ex["metadata"]["image_doc_ids"],
                    image_metadata
                )

                if not images:
                    continue
                
                # Build caption pool
                texts = []

                for img_id in image_ids:
                    if img_id in image_metadata and image_metadata[img_id].get("caption"):
                        texts.append(image_metadata[img_id]["caption"])

                injected_poison = None

                if USE_POISONED_CAPTIONS and poisoned_metadata is not None:
                    for img_id in image_ids:
                        if img_id not in poisoned_metadata:
                            continue

                        for poison_entry in poisoned_metadata[img_id].get("poisoned", []):
                            if poison_entry["query"] == ex["question"]:
                                injected_poison = poison_entry["poisoned_candidates"][0]
                                texts.append(injected_poison)
                                break

                        if injected_poison is not None:
                            break

                if not texts:
                    continue

                # Run RAG
                try:
                    output = rag.generate(
                        question=question,
                        images=images,
                        texts=texts,
                        max_new_tokens=150
                    )
                except Exception as e:
                    results.append({
                        "qid": ex.get("qid"),
                        "question": question,
                        "error": str(e)
                    })
                    print("error occurred!")
                    print(str(e))
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

                    # Retrieval results
                    "retrieved_image_ids": retrieved_image_ids,
                    "retrieved_captions": retrieved_captions,
                    "image_scores": output["image_scores"],
                    "text_scores": output["text_scores"],

                    # Attack bookkeeping
                    "poison_injected": injected_poison is not None,
                    "poison_caption": injected_poison,
                })

            print(f"Saving results to {output_file}")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)

            print("Done with this configuration.")

    print("\nAll runs completed.")


if __name__ == "__main__":
    main()
