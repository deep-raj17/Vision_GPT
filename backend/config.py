"""Path and model configuration for Vision GPT."""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "satellite_caption_dataset"
RAW_IMAGE_DIR = DATA_DIR / "raw_images"
STAGE1_CHECKPOINT = (
    ROOT
    / "vision_gpt_backend"
    / "Output"
    / "stage1_alignment"
    / "checkpoint-800"
    / "checkpoint-800"
)
STAGE1_CONFIG = STAGE1_CHECKPOINT / "config.json"

# Fast, stable BLIP base. Keep this fixed for the caption pipeline.
MODEL_ID = "Salesforce/blip-image-captioning-base"
CAPTION_MODEL_ID = MODEL_ID

DETAILED_CAPTION_PROMPT = (
    "Describe this satellite image in detail. Include land cover, buildings, roads, "
    "water bodies, vegetation, agricultural fields, terrain, spatial arrangement, "
    "visible patterns, and possible human activity."
)

BLIP_CONDITIONAL_PROMPT = os.getenv("BLIP_CONDITIONAL_PROMPT", "a satellite image of")

MAX_NEW_TOKENS = 35
MIN_NEW_TOKENS = 8
NUM_BEAMS = 3
REPETITION_PENALTY = 1.4
NO_REPEAT_NGRAM_SIZE = 3

BLIP_GENERATION_CONFIG = {
    "max_new_tokens": MAX_NEW_TOKENS,
    "min_new_tokens": MIN_NEW_TOKENS,
    "num_beams": NUM_BEAMS,
    "repetition_penalty": REPETITION_PENALTY,
    "length_penalty": float(os.getenv("BLIP_LENGTH_PENALTY", "1.0")),
    "no_repeat_ngram_size": NO_REPEAT_NGRAM_SIZE,
    "early_stopping": os.getenv("BLIP_EARLY_STOPPING", "true").lower() == "true",
}

MIN_DETAILED_WORDS = int(os.getenv("MIN_DETAILED_WORDS", "200"))
MAX_DETAILED_WORDS = int(os.getenv("MAX_DETAILED_WORDS", "300"))
MAX_INFERENCE_IMAGE_SIDE = int(os.getenv("MAX_INFERENCE_IMAGE_SIDE", "1024"))

GENERATION_CONFIG = {
    "max_new_tokens": int(os.getenv("CAPTION_MAX_NEW_TOKENS", "80")),
    "min_new_tokens": int(os.getenv("CAPTION_MIN_NEW_TOKENS", "30")),
    "num_beams": int(os.getenv("CAPTION_NUM_BEAMS", "5")),
    "repetition_penalty": float(os.getenv("CAPTION_REPETITION_PENALTY", "1.2")),
    "length_penalty": float(os.getenv("CAPTION_LENGTH_PENALTY", "1.4")),
    "no_repeat_ngram_size": int(os.getenv("CAPTION_NO_REPEAT_NGRAM_SIZE", "3")),
}


def load_stage1_config() -> dict:
    if STAGE1_CONFIG.is_file():
        with STAGE1_CONFIG.open(encoding="utf-8") as handle:
            return json.load(handle)
    return {}
