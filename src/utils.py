import os
import json
import gzip
from PIL import Image, UnidentifiedImageError


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


def load_images(IMAGE_DIR, image_ids):
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