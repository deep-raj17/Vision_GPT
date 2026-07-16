"""Download the fixed BLIP-base model used by Vision_GPT."""

from __future__ import annotations

import os

for key in [
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "http_proxy",
    "https_proxy",
    "ALL_PROXY",
    "all_proxy",
]:
    os.environ.pop(key, None)

os.environ["NO_PROXY"] = "localhost,127.0.0.1,huggingface.co,cdn-lfs.huggingface.co"
os.environ["no_proxy"] = "localhost,127.0.0.1,huggingface.co,cdn-lfs.huggingface.co"

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
HF_CACHE = os.path.join(PROJECT_ROOT, "models", "huggingface")
os.makedirs(HF_CACHE, exist_ok=True)
os.environ.setdefault("HF_HOME", HF_CACHE)
os.environ.setdefault("HF_HUB_CACHE", os.path.join(HF_CACHE, "hub"))

from transformers import BlipForConditionalGeneration, BlipProcessor

MODEL_ID = "Salesforce/blip-image-captioning-base"

print("Downloading processor...")
BlipProcessor.from_pretrained(MODEL_ID)

print("Downloading model...")
BlipForConditionalGeneration.from_pretrained(MODEL_ID)

print("Model downloaded successfully.")
