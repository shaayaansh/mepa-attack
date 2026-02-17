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



---

## Running the Multimodal RAG Pipeline

The main RAG pipeline is implemented in `src/run_rag.py`.  
It supports clean evaluation, metadata poisoning, and robustness testing via
command-line flags.

All runs are executed from the repository root using:

```bash
python -m src.run_rag [FLAGS]
```

### Available Flags

- `--use_poison`  
  Injects query-specific poisoned captions into the candidate caption pool.

- `--robustness`  
  Replaces original user queries with paraphrased versions (for robustness evaluation).

- `--k <int>`  
  Sets the number of retrieved images and captions (`top_k_images = top_k_texts = k`).

By default:
- `use_poison = False`
- `robustness = False`
- `k = 1`

---

### Clean Baseline

```bash
python -m src.run_rag --k 1
```

outputs are saved to

```bash
results/clean_results/
```

### Poisoned setting

```bash
python -m src.run_rag --use_poison --k 1
```

### Robustness Evaluation setting (using paraphrased queries)

```bash
python -m src.run_rag --use_poison --robustness --k 1
```


## Evaluating RAG Performance

After running the RAG pipeline, evaluate the generated results using the
evaluation script. The results file is automatically inferred from the dataset
name and whether the run is clean or poisoned.

From the repository root, run:

```bash
python src/eval_rag.py --dataset_name <mmqa|webqa> --split <clean|poisoned>
```

This script will:

1. Load the RAG output JSON file
2. Compare model-generated answers against gold answers
3. Compute exact-match (EM) accuracy
4. Report the total number of evaluated questions and skipped examples

The evaluation output is printed directly to the terminal and can be used to
compare clean versus poisoned RAG performance.



## Evaluating MEPA Attacks (Retrieval, Answer, and Cohesion Metrics)

In addition to standard RAG accuracy evaluation, we provide a unified evaluation
script for **MEPA-style multimodal poisoning attacks**. This script measures
whether poisoned evidence is retrieved, whether the model adopts the attacker’s
narrative, and whether simple image–text consistency defenses can detect the
attack.

The evaluation supports **both MMQA and WebQA**, automatically handling
dataset-specific schema differences.

---

### Running the MEPA Evaluation

From the repository root, run:

```bash
python src/eval_mepa_attack.py <RESULTS_JSON> \
  --k 3 \
  --defense_threshold 0.7 \
  --image_root <PATH_TO_IMAGE_DIRECTORY>
```

### Example
```bash
python src/eval_mepa_attack.py \
  results/rag_clip_llava_mmqa_poisoned.json \
  --k 3 \
  --defense_threshold 0.7 \
  --image_root datasets/mmqa/final_dataset_images
```


### Metrics Reported

The evaluation script reports three groups of metrics:

---

#### A) Retrieval Metrics

- **ROrig@k**  
  Retrieval recall of original (gold) evidence: the fraction of questions for
  which at least one gold image appears in the top-k retrieved images.  
  *(Defined only for MMQA, where gold image supervision is available.)*

- **RPois@k**  
  Retrieval recall of poisoned evidence: the fraction of poisoned questions for
  which the adversarial (poisoned) caption appears in the top-k retrieved
  captions.

---

#### B) Answer Metrics

- **ACCOrig_EM**  
  Exact-match accuracy of the model’s answer against the gold answer(s).

- **ACCPois_ASR (Attack Success Rate)**  
  Fraction of poisoned questions for which the model’s answer adopts the
  attacker’s narrative, measured via string matching against the injected
  poison caption.

---

#### C) Cohesion / Detection Metrics

- **Mean_Image_Metadata_Sim**  
  Mean cosine similarity between CLIP image embeddings and CLIP text embeddings
  for retrieved (image, caption) pairs in poisoned examples.

- **DetectionRate@τ**  
  Fraction of poisoned examples flagged by a simple image–text consistency
  detector that marks an item as suspicious if the CLIP cosine similarity is
  below a threshold τ (e.g., 0.2).




