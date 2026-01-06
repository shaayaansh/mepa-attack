# MEPA-Attack

MEPA-Attack is a research codebase for studying attacks on multimodal
Retrieval-Augmented Generation (RAG) systems.

The framework supports multimodal retrievers (e.g., CLIP) and generators
(e.g., LLaVA, Qwen-VL) and is designed for controlled experimentation on
datasets such as MMQA and WebQA.

---

## Repository Structure

```text
mepa-attack/
├── src/                # Core RAG components (retriever, generator, RAGModel)
├── datasets/           # In-repo dataset organization (MMQA, WebQA, etc.)
├── results/            # Generated RAG outputs (not committed)
├── download_images.py
└── README.md
```

---

## Dataset Setup (MMQA)

### 1. Download MMQA images

From the repository root, run:

```bash
python download_images.py
```


This script will download and extract the MMQA image archive and store the
images at:

```bash
datasets/mmqa/final_dataset_images/
```

After setup, the MMQA dataset directory should look like:

```text
datasets/mmqa/
├── MMQA_train_image_text_only.jsonl.gz
├── MMQA_texts.jsonl.gz
└── final_dataset_images/
    ├── <image_id>.jpg
    └── ...
```


## Generating Metadata poisoning attack
### Overview
We model metadata poisoning by generating malicious image captions
that:
remain visually plausible for a given image,
are semantically relevant to a user query,
contradict the true answer, and
are indexed by the retriever like normal metadata.
The attack is query-specific and does not modify images or models.

### Intended behavior 

For each RAG query, we construct a candidate text pool as follows:

```python
Before (clean):
texts = [
    clean_caption(img_1),
    clean_caption(img_2),
    clean_caption(img_3)
]

After (poisoned):
texts = [
    clean_caption(img_1),
    clean_caption(img_2),
    clean_caption(img_3),
    poisoned_caption(img_k)   # exactly ONE injected caption
]
```

Only one poisoned caption is injected per query, corresponding to a
gold-supporting image.

---

### Running the Attack Generator

From the repository root:
```python
python -m src.generate_attack
```

This script will:

1. Load MMQA test questions and gold image supports
2. Generate query-specific poisoned captions by conditioning on:
   - the image’s clean caption,
   - the user query,
   - the ground-truth answer (to generate a plausible contradiction)
3. Clean and filter generated captions
4. Drop failed generations
5. Save the poisoned metadata to:

```text
datasets/mmqa-mmpoisonrag/MMQA_image_metadata_poisoned.json
```

Each entry is keyed by image ID and may contain multiple poisoned captions
corresponding to different queries.

