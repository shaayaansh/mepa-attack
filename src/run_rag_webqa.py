import os
import json
from tqdm import tqdm
from PIL import Image
import argparse
from src.retriever import Retriever
from src.generator import Generator
from src.rag_model import RAGModel


# =========================
# Environment / cache
# =========================

os.environ["DISABLE_TRANSFORMERS_STREAMING"] = "1"
os.environ["HF_HOME"] = "/scratch/shayan/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "/scratch/shayan/hf_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/scratch/shayan/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/scratch/shayan/hf_cache"


CACHE_DIR = "/scratch/shayan/hf_cache"



# paths
DATASET_NAME = "webqa-mmpoisonrag"
DATASET_ROOT = f"datasets/{DATASET_NAME}"

DATA_PATH = f"{DATASET_ROOT}/WebQA_test_image.json"
IMAGE_DIR = f"{DATASET_ROOT}/extracted_images"
POISONED_METADATA_PATH = f"{DATASET_ROOT}/WebQA_image_metadata_poisoned.json"

PARAPHRASE_PATH = (
    "/scratch/shayan/Projects/mepa-attack/results/paraphrased_questions/"
    "webqa_paraphrased_questions.json"
)

CLEAN_RESULTS_DIR = "/scratch/shayan/Projects/mepa-attack/results/clean_results"
POISON_RESULTS_DIR = "/scratch/shayan/Projects/mepa-attack/results/webqa_results"
ROBUSTNESS_RESULTS_DIR = "/scratch/shayan/Projects/mepa-attack/results/robustness_results"

os.makedirs(CLEAN_RESULTS_DIR, exist_ok=True)
os.makedirs(POISON_RESULTS_DIR, exist_ok=True)
os.makedirs(ROBUSTNESS_RESULTS_DIR, exist_ok=True)



# Model grids
RETRIEVERS = {
    "clip": "openai/clip-vit-base-patch32",
    "openclip": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    "sigclip": "google/siglip-base-patch16-224",
    "flava": "facebook/flava-full",
}

GENERATORS = {
    "llava": "llava-hf/llava-1.5-7b-hf",
    "blip2": "Salesforce/blip2-flan-t5-xl",
}


# =========================
# Utilities
# =========================

def load_webqa_image(image_id):
    path = os.path.join(IMAGE_DIR, f"{image_id}.jpg")
    return Image.open(path).convert("RGB")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--use_poison", action="store_true",
                        help="Use poisoned captions")
    parser.add_argument("--robustness", action="store_true",
                        help="Use paraphrased queries")
    parser.add_argument("--k", type=int, default=2)

    args = parser.parse_args()
    USE_POISONED_CAPTIONS = args.use_poison
    ROBUSTNESS_MODE = args.robustness
    K = args.k

    poisoned_metadata = None
    if USE_POISONED_CAPTIONS:
        print("Loading poisoned metadata...")
        with open(POISONED_METADATA_PATH, "r") as f:
            poisoned_metadata = json.load(f)

    print("Loading WebQA data...")
    with open(DATA_PATH, "r") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} WebQA test examples")

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
                f"{save_dir}/rag_{retriever_type}_{generator_type}_webqa_{suffix}_k={K}.json"
            )

            results = []

            print("Running RAG inference...")
            for guid, ex in tqdm(data.items()):

                original_question = ex["Q"].strip('"')

                if ROBUSTNESS_MODE:
                    question = paraphrase_map.get(guid, original_question)
                else:
                    question = original_question


                # Long-form answers
                gold_texts = [a.strip('"') for a in ex.get("A", [])]

                # Official EM answer
                gold_em = ex.get("EM")

                # Gold image IDs (positive facts only)
                pos_facts = ex.get("img_posFacts", [])
                image_instances = [
                    {
                        "doc_id": str(fact["image_id"]),
                        "doc_part": "image"
                    }
                    for fact in pos_facts
                ]

                # Structured gold_answers (MMQA-style format)
                gold_answers = [
                    {
                        "answer": gold_text,
                        "EM": gold_em,
                        "type": "string",
                        "modality": "image",
                        "text_instances": [],
                        "table_indices": [],
                        "image_instances": image_instances
                    }
                    for gold_text in gold_texts
                ]


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

                # -------------------------
                # Inject poisoned caption
                # -------------------------
                injected_poison = None

                if USE_POISONED_CAPTIONS and poisoned_metadata is not None:
                    q_norm = " ".join(original_question.split()).strip().lower()

                    for img_id in image_ids:
                        key = str(img_id)

                        if key not in poisoned_metadata:
                            continue

                        for poison_entry in poisoned_metadata[key].get("poisoned", []):
                            pq = poison_entry.get("query", "")
                            pq_norm = " ".join(pq.split()).strip().lower()

                            if pq_norm == q_norm:
                                injected_poison = poison_entry["poisoned_candidates"][0]
                                texts.append(injected_poison)
                                break

                        if injected_poison is not None:
                            break

                # -------------------------
                # Run RAG
                # -------------------------
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
                    "original_question": original_question,
                    "model_answer": output["answer"],
                    "gold_answers": gold_answers,
                    # TODO: add EM gold answers to results
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

    print("\nAll WebQA runs completed.")


if __name__ == "__main__":
    main()
