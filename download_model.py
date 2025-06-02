import os
import requests

MODEL_URL = "https://huggingface.co/username/model-repo/resolve/main/yolo11x.pt"
MODEL_PATH = "yolo11x.pt"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        response = requests.get(MODEL_URL)
        response.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)
        print("Download complete.")
