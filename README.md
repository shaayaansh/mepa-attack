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

datasets/mmqa/
├── MMQA_train_image_text_only.jsonl.gz
├── MMQA_texts.jsonl.gz
└── final_dataset_images/
    ├── <image_id>.jpg
    └── ...
