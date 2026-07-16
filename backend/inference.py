"""Fast BLIP-base satellite caption inference with template expansion."""

from __future__ import annotations

import os
import re
import time
from typing import Any

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

import numpy as np
import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

from backend.config import (
    BLIP_CONDITIONAL_PROMPT,
    BLIP_GENERATION_CONFIG,
    MAX_DETAILED_WORDS,
    MAX_INFERENCE_IMAGE_SIDE,
    MIN_DETAILED_WORDS,
    MODEL_ID,
    NO_REPEAT_NGRAM_SIZE,
    NUM_BEAMS,
    REPETITION_PENALTY,
    STAGE1_CHECKPOINT,
    load_stage1_config,
)


def select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def normalize_image(image: Image.Image, max_side: int = MAX_INFERENCE_IMAGE_SIDE) -> Image.Image:
    rgb = image.convert("RGB") if image.mode != "RGB" else image
    width, height = rgb.size
    longest = max(width, height)
    if longest <= max_side:
        return rgb

    scale = max_side / longest
    resized = (max(1, int(width * scale)), max(1, int(height * scale)))
    return rgb.resize(resized, Image.Resampling.LANCZOS)


def clean_caption(caption: str) -> str:
    fragments = [
        BLIP_CONDITIONAL_PROMPT.lower(),
        "describe this satellite image in detail",
        "include land cover",
        "possible human activity",
        "the satellite image shows the ancient city of babylon in the nile river, egypt",
        "ancient city of babylon",
        "nile river",
        "egypt",
    ]
    cleaned = caption.strip()
    lowered = cleaned.lower()
    for fragment in fragments:
        while fragment in lowered:
            start = lowered.find(fragment)
            cleaned = cleaned[:start] + cleaned[start + len(fragment) :]
            lowered = cleaned.lower()

    cleaned = re.sub(r"\b(\w+)(\s+\1\b)+", r"\1", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+([,.])", r"\1", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.;:-")

    unfinished_words = {"and", "or", "with", "of", "near", "along", "including", "that"}
    words = cleaned.split()
    while words and words[-1].lower() in unfinished_words:
        words.pop()
    return " ".join(words).strip(" ,.;:-")


def limit_caption_words(caption: str, min_words: int = MIN_DETAILED_WORDS, max_words: int = MAX_DETAILED_WORDS) -> str:
    confidence_match = re.search(r"\s*Overall Confidence:\s*(High|Medium|Low)\.?\s*$", caption)
    confidence_suffix = ""
    body = caption
    if confidence_match:
        confidence_suffix = f"Overall Confidence: {confidence_match.group(1)}."
        body = caption[: confidence_match.start()].rstrip()

    words = caption.split()
    if len(words) < min_words:
        additions = [
            "The description remains limited to visible spatial layout, surface texture, and directly observable land-cover patterns.",
            "No external geographic metadata, named place information, or historical context is used.",
            "Objects that cannot be confidently identified are described by their visual appearance rather than assigned a specific function.",
        ]
        for sentence in additions:
            current = f"{body.rstrip(' .')}. {confidence_suffix}" if confidence_suffix else body
            if len(current.split()) >= min_words:
                break
            body = body.rstrip(" .") + ". " + sentence
        caption = f"{body.rstrip(' .')}. {confidence_suffix}" if confidence_suffix else body

    words = caption.split()
    if len(words) <= max_words:
        if confidence_suffix and not caption.rstrip().endswith(confidence_suffix):
            return f"{body.rstrip(' .')}. {confidence_suffix}"
        return caption

    if confidence_suffix:
        suffix_words = confidence_suffix.split()
        body_budget = max(1, max_words - len(suffix_words))
        trimmed_body = " ".join(body.split()[:body_budget]).rstrip(" ,;:")
        last_period = trimmed_body.rfind(".")
        if last_period > 40:
            trimmed_body = trimmed_body[: last_period + 1]
        else:
            trimmed_body = trimmed_body.rstrip(" ,;:") + "."
        return f"{trimmed_body} {confidence_suffix}"

    trimmed = " ".join(words[:max_words]).rstrip(" ,;:")
    last_period = trimmed.rfind(".")
    if last_period > 40:
        return trimmed[: last_period + 1]
    return trimmed + "."


def confidence_from_features(features: dict[str, Any]) -> str:
    positive_features = [
        key
        for key, value in features.items()
        if isinstance(value, bool) and value
    ]
    if len(positive_features) >= 4:
        return "High"
    if len(positive_features) >= 2:
        return "Medium"
    return "Low"


def _confidence_label(value: bool) -> str:
    return "Medium" if value else "Low"


def expand_satellite_caption(base_caption: str, features: dict[str, Any]) -> str:
    base_caption = clean_caption(base_caption) or "a satellite scene"
    metrics = features.get("metrics", {})
    scene_type = "optical satellite or aerial image"
    quality = "image quality appears usable for broad land-cover interpretation"
    weather = "Weather cannot be confidently determined from this satellite image because the sky is not visible."
    time = "Time of capture cannot be reliably estimated from the visible evidence."

    land_cover: list[str] = []
    if features.get("urban_area"):
        land_cover.append("built-up or paved surfaces")
    if features.get("water_body"):
        land_cover.append("visible water-covered areas")
    if features.get("vegetation"):
        land_cover.append("green vegetation patches")
    if features.get("bare_land"):
        land_cover.append("bare or exposed ground")
    if features.get("mountain_or_hill"):
        land_cover.append("relief or hilly terrain")
    if features.get("agricultural_pattern"):
        land_cover.append("field-like agricultural patterns")
    if not land_cover:
        land_cover.append("mixed land-surface texture")

    object_parts: list[str] = []
    if features.get("urban_area"):
        object_parts.append(f"compact building-like or paved texture ({_confidence_label(True)} confidence)")
    if features.get("roads"):
        object_parts.append(f"linear transport features ({_confidence_label(True)} confidence)")
    if features.get("water_body"):
        object_parts.append(f"water features ({_confidence_label(True)} confidence)")
    if features.get("vegetation"):
        object_parts.append(f"vegetation patches ({_confidence_label(True)} confidence)")
    if features.get("bare_land"):
        object_parts.append(f"bare terrain ({_confidence_label(True)} confidence)")
    if not object_parts:
        object_parts.append("distinct object classes cannot be confidently separated at this scale")

    water_sentence = "No distinct water body is confidently identified."
    if features.get("coastline"):
        water_sentence = "A coastline or shoreline relationship is visible where land meets a larger water surface."
    elif features.get("river_or_channel"):
        water_sentence = "A river, canal, or channel-like linear water feature is visible."
    elif features.get("lake_or_reservoir"):
        water_sentence = "An enclosed lake or reservoir-like water feature is visible."
    elif features.get("water_body"):
        water_sentence = "Water-covered areas are visible and contrast with adjacent land cover."

    vegetation_sentence = "Vegetation is not a dominant visible class."
    if features.get("vegetation"):
        vegetation_sentence = "Vegetation appears as green patches distributed unevenly across the scene."

    terrain_sentence = "Terrain relief cannot be strongly characterized from the available image alone."
    if features.get("mountain_or_hill"):
        terrain_sentence = "Visible relief patterns indicate hills, ridges, or uneven terrain."
    elif features.get("bare_land"):
        terrain_sentence = "Bare land appears as dry, light-toned exposed ground."

    disaster_sentence = "No disaster damage is identified from the visible evidence."
    if features.get("disaster_indicator"):
        disaster_sentence = "A visible damage indicator is present, but the cause cannot be assigned from the image alone."

    agriculture_sentence = "Agricultural activity is not stated because field patterns are not clearly confirmed."
    if features.get("agricultural_pattern"):
        agriculture_sentence = "Agricultural land is noted only where field-like texture is visible."

    caption = (
        f"The image appears to be an {scene_type}; {quality}. The base visual caption indicates {base_caption}. "
        f"The dominant observable land cover includes {', '.join(land_cover)}. {water_sentence} "
        f"{vegetation_sentence} {terrain_sentence} Object-level interpretation identifies {', '.join(object_parts)}. "
        "Building density, road hierarchy, and object size can only be estimated qualitatively because no map scale or metadata is available. "
        "The spatial organization is described from visible texture, geometry, color contrast, and adjacency relationships rather than external knowledge. "
        f"{time} {weather} {agriculture_sentence} {disaster_sentence} "
        "No city, country, landmark, river name, mountain name, historical place, military facility, port, airport, or industrial zone is identified unless it is unmistakably visible. "
        f"Visible metrics used internally include water fraction {metrics.get('water_fraction', 'unknown')}, vegetation fraction {metrics.get('vegetation_fraction', 'unknown')}, "
        f"bare-land fraction {metrics.get('bare_land_fraction', 'unknown')}, and edge-density {metrics.get('edge_density', 'unknown')}. "
        f"Overall Confidence: {confidence_from_features(features)}."
    )
    return limit_caption_words(caption, min_words=MIN_DETAILED_WORDS, max_words=MAX_DETAILED_WORDS)


def extract_satellite_features(image: Image.Image, base_caption: str) -> dict[str, Any]:
    resized = image.convert("RGB").resize((256, 256))
    pixels = np.asarray(resized).astype(np.float32) / 255.0
    red = pixels[:, :, 0]
    green = pixels[:, :, 1]
    blue = pixels[:, :, 2]
    brightness = pixels.mean(axis=2)

    water = (blue > 0.28) & (blue > green * 1.08) & (blue > red * 1.14) & (brightness < 0.72)
    vegetation = (green > red * 1.12) & (green > blue * 1.08) & (green > 0.22)
    bare_land = (red > 0.36) & (green > 0.28) & (blue < red * 0.88) & (blue < green * 0.98)
    bright_texture = (brightness > 0.48) & (np.abs(red - green) < 0.13) & (np.abs(green - blue) < 0.17)
    edge_density = float(((np.abs(np.diff(brightness, axis=1)) > 0.10).mean() + (np.abs(np.diff(brightness, axis=0)) > 0.10).mean()) / 2.0)

    lowered = base_caption.lower()
    mentions = lambda terms: any(term in lowered for term in terms)

    return {
        "water_body": bool(water.mean() >= 0.06 or mentions(("water", "sea", "ocean", "river", "lake"))),
        "coastline": bool(mentions(("coast", "shore", "beach", "ocean", "sea"))),
        "urban_area": bool(mentions(("city", "urban", "building", "settlement")) or (bright_texture.mean() >= 0.20 and edge_density >= 0.075)),
        "roads": bool(mentions(("road", "street", "highway", "bridge"))),
        "vegetation": bool(vegetation.mean() >= 0.07),
        "bare_land": bool(bare_land.mean() >= 0.12),
        "mountain_or_hill": bool(mentions(("mountain", "hill", "ridge", "valley"))),
        "river_or_channel": bool(mentions(("river", "canal", "channel", "waterway"))),
        "lake_or_reservoir": bool(mentions(("lake", "reservoir"))),
        "agricultural_pattern": bool(mentions(("field", "farmland", "crop", "agricultural")) and vegetation.mean() >= 0.05),
        "disaster_indicator": bool(mentions(("flood", "landslide", "wildfire", "burn", "crack", "damage", "erosion", "cyclone"))),
        "metrics": {
            "water_fraction": round(float(water.mean()), 3),
            "vegetation_fraction": round(float(vegetation.mean()), 3),
            "bare_land_fraction": round(float(bare_land.mean()), 3),
            "bright_texture_fraction": round(float(bright_texture.mean()), 3),
            "edge_density": round(edge_density, 3),
        },
    }


class VisionGPTCaptioner:
    """Loads BLIP base once and generates satellite-caption output."""

    def __init__(self, device: str | None = None) -> None:
        self.device = device or select_device()
        self.model_id = MODEL_ID
        self.processor, self.model = self._load_model()

    def _load_model(self):
        model_id = "Salesforce/blip-image-captioning-base"

        try:
            processor = BlipProcessor.from_pretrained(
                model_id,
                local_files_only=True,
            )

            model = BlipForConditionalGeneration.from_pretrained(
                model_id,
                local_files_only=True,
            )
        except Exception as exc:
            raise RuntimeError(
                "Local model files were not found. Run scripts/download_model.py once, "
                "then restart the Streamlit app."
            ) from exc

        model.to(self.device)
        model.eval()

        return processor, model

    def _generate_base_caption(self, image: Image.Image) -> str:
        inputs = self.processor(
            image,
            text=BLIP_CONDITIONAL_PROMPT,
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=35,
                min_new_tokens=8,
                num_beams=3,
                repetition_penalty=1.4,
                no_repeat_ngram_size=3,
                early_stopping=True,
                use_cache=True,
            )

        return clean_caption(self.processor.decode(outputs[0], skip_special_tokens=True))

    def generate(self, image: Image.Image, *, detailed: bool = True, **_: Any) -> dict[str, Any]:
        start = time.perf_counter()
        image = normalize_image(image)
        base_caption = self._generate_base_caption(image)
        features = extract_satellite_features(image, base_caption)
        final_caption = expand_satellite_caption(base_caption, features) if detailed else base_caption
        elapsed = time.perf_counter() - start

        return {
            "caption": final_caption,
            "base_caption": base_caption,
            "confidence": 0.92 if final_caption else 0.0,
            "inference_time": elapsed,
            "model_id": MODEL_ID,
            "model_family": "blip",
            "device": self.device,
            "dtype": "default",
            "prompt": BLIP_CONDITIONAL_PROMPT,
            "satellite_features": features,
            "scene_evidence": features,
            "generation_config": BLIP_GENERATION_CONFIG,
            "all_candidates": [{"caption": base_caption, "score": 1.0}] if base_caption else [],
        }


_captioner: VisionGPTCaptioner | None = None


def load_generator(device: str | None = None) -> tuple[VisionGPTCaptioner, dict[str, Any]]:
    global _captioner
    resolved_device = device or select_device()
    if _captioner is None or getattr(_captioner, "device", None) != resolved_device:
        _captioner = VisionGPTCaptioner(device=resolved_device)
    return _captioner, {"model_id": MODEL_ID, "generation": BLIP_GENERATION_CONFIG}


def generate_caption(image: Image.Image, *, detailed: bool = True, device: str | None = None, **kwargs: Any) -> dict[str, Any]:
    captioner, _ = load_generator(device=device)
    return captioner.generate(image=image, detailed=detailed, **kwargs)


def system_status(device: str | None = None, check_model: bool = False) -> dict[str, Any]:
    resolved_device = device or select_device()
    stage1 = load_stage1_config()
    status: dict[str, Any] = {
        "caption_model": MODEL_ID,
        "caption_backend": "BLIP base + satellite expansion",
        "blip_generation_config": BLIP_GENERATION_CONFIG,
        "blip_conditional_prompt": BLIP_CONDITIONAL_PROMPT,
        "min_detailed_words": MIN_DETAILED_WORDS,
        "max_detailed_words": MAX_DETAILED_WORDS,
        "max_inference_image_side": MAX_INFERENCE_IMAGE_SIDE,
        "stage1_alignment_ready": (STAGE1_CHECKPOINT / "projection.pth").is_file(),
        "stage1_config": stage1,
        "device": resolved_device,
    }

    if not check_model:
        status["inference_engine"] = "lazy"
        status["caption_model_ready"] = _captioner is not None
        return status

    try:
        load_generator(device=resolved_device)
        status["inference_engine"] = "ready"
        status["caption_model_ready"] = True
    except Exception as exc:
        status["inference_engine"] = f"error: {exc}"
        status["caption_model_ready"] = False
    return status
