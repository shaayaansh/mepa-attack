import os
import urllib.request
import zipfile
import shutil

MMQA_IMAGE_URL = (
    "https://multimodalqa-images.s3-us-west-2.amazonaws.com/"
    "final_dataset_images/final_dataset_images.zip"
)

DATASET_DIR = "datasets/mmqa"
ZIP_PATH = os.path.join(DATASET_DIR, "final_dataset_images.zip")
IMAGE_DIR = os.path.join(DATASET_DIR, "final_dataset_images")


def download_images():
    os.makedirs(DATASET_DIR, exist_ok=True)

    # If images already exist, skip
    if os.path.exists(IMAGE_DIR) and len(os.listdir(IMAGE_DIR)) > 0:
        print(f"Images already exist at {IMAGE_DIR}, skipping download.")
        return

    print("Downloading MMQA images...")
    urllib.request.urlretrieve(MMQA_IMAGE_URL, ZIP_PATH)
    print("Download complete.")

    print("Extracting images...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(DATASET_DIR)

    print("Extraction complete.")

    # Cleanup
    os.remove(ZIP_PATH)
    print("Cleaned up zip file.")

    print(f"Images are available at: {IMAGE_DIR}")


if __name__ == "__main__":
    download_images()
