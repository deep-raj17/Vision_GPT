"""Path configuration for Vision GPT frontend."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
STAGE1_CHECKPOINT = (
    ROOT
    / "vision_gpt_backend"
    / "Output"
    / "stage1_alignment"
    / "checkpoint-800"
    / "checkpoint-800"
)
STAGE1_CONFIG = STAGE1_CHECKPOINT / "config.json"

# Pretrained BLIP model — produces real English captions (no UNK tokens)
CAPTION_MODEL_ID = "Salesforce/blip-image-captioning-base"


def load_stage1_config() -> dict:
    if STAGE1_CONFIG.is_file():
        with STAGE1_CONFIG.open(encoding="utf-8") as handle:
            return json.load(handle)
    return {}
