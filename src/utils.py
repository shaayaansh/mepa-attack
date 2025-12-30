import os
import json
import gzip
from PIL import Image, UnidentifiedImageError


def load_mmqa_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_text_corpus(path):
    corpus = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            corpus[ex["id"]] = ex["text"]
    return corpus


def load_images_from_metadata(image_dir, image_doc_ids, image_metadata):
    images = []
    valid_ids = []

    for img_id in image_doc_ids:
        meta = image_metadata.get(img_id)
        if meta is None:
            continue

        img_path = os.path.join(image_dir, meta["path"])
        if not os.path.exists(img_path):
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except (UnidentifiedImageError, OSError) as e:
            print(f"[WARN] Skipping image {img_path}: {e}")
            continue

        images.append(img)
        valid_ids.append(img_id)

    return images, valid_ids
