"""Vision GPT - satellite image detailed captioning web app."""

from __future__ import annotations

import os
import sys
import hashlib
from io import BytesIO
from pathlib import Path

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

ROOT = Path(__file__).resolve().parent
HF_CACHE = ROOT / "models" / "huggingface"
HF_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(HF_CACHE))
os.environ.setdefault("HF_HUB_CACHE", str(HF_CACHE / "hub"))

import pandas as pd
import streamlit as st
from PIL import Image

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config import (
    BLIP_CONDITIONAL_PROMPT,
    BLIP_GENERATION_CONFIG,
    CAPTION_MODEL_ID,
    DETAILED_CAPTION_PROMPT,
    GENERATION_CONFIG,
)
from backend.inference import load_generator, normalize_image, select_device, system_status

st.set_page_config(
    page_title="Vision GPT",
    page_icon="VIS",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-title { font-size: 2.2rem; font-weight: 700; margin-bottom: 0.2rem; }
    .subtitle { color: #8b8ba7; margin-bottom: 1.5rem; }
    .status-ok { color: #4ade80; font-weight: 600; }
    .status-bad { color: #f87171; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

if "history" not in st.session_state:
    st.session_state.history = []
if "generator_bundle" not in st.session_state:
    st.session_state.generator_bundle = None
if "caption_result_cache" not in st.session_state:
    st.session_state.caption_result_cache = {}
if "active_caption_key" not in st.session_state:
    st.session_state.active_caption_key = None
if "history_keys" not in st.session_state:
    st.session_state.history_keys = set()


@st.cache_resource(show_spinner=False)
def cached_generator(device: str):
    return load_generator(device=device)


def get_generator(device: str):
    bundle = st.session_state.generator_bundle
    if bundle and getattr(bundle[0], "device", None) == device:
        return bundle

    with st.spinner("Loading model once..."):
        bundle = cached_generator(device)
    st.session_state.generator_bundle = bundle
    return bundle


def is_blip_model() -> bool:
    normalized = CAPTION_MODEL_ID.lower()
    return all(marker not in normalized for marker in ("blip2", "blip-2", "llava"))


def render_header() -> None:
    st.markdown(
        '<div class="main-title">Satellite Image Detailed Captioning and Scene Understanding</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="subtitle">Upload satellite imagery and generate a grounded visual interpretation.</div>',
        unsafe_allow_html=True,
    )
    st.warning("AI captions should be verified with geospatial metadata for professional use.")


def render_sidebar() -> dict:
    st.sidebar.header("Inference Settings")
    device = select_device()
    st.sidebar.caption(f"Device: **{device.upper()}**")
    st.sidebar.caption(f"Model: `{CAPTION_MODEL_ID}`")

    detailed = st.sidebar.checkbox("Use detailed satellite interpretation", value=True)

    if is_blip_model():
        prompt = BLIP_CONDITIONAL_PROMPT
        beam_size = BLIP_GENERATION_CONFIG["num_beams"]
        min_new_tokens = BLIP_GENERATION_CONFIG["min_new_tokens"]
        max_new_tokens = BLIP_GENERATION_CONFIG["max_new_tokens"]
        temperature = 1.0
        length_penalty = BLIP_GENERATION_CONFIG["length_penalty"]
        top_k = 0
        st.sidebar.caption(f"BLIP prompt: `{BLIP_CONDITIONAL_PROMPT}`")
        st.sidebar.json(BLIP_GENERATION_CONFIG)
    else:
        beam_size = st.sidebar.slider("Beam size", 1, 10, GENERATION_CONFIG["num_beams"])
        min_new_tokens = st.sidebar.slider("Min new tokens", 5, 80, GENERATION_CONFIG["min_new_tokens"])
        max_new_tokens = st.sidebar.slider("Max new tokens", 30, 160, GENERATION_CONFIG["max_new_tokens"])
        temperature = st.sidebar.slider("Temperature", 0.1, 2.0, 1.0, 0.1)
        length_penalty = st.sidebar.slider(
            "Length penalty",
            0.0,
            2.0,
            GENERATION_CONFIG["length_penalty"],
            0.05,
        )
        top_k = st.sidebar.number_input("Top-K (0 = beam/greedy)", min_value=0, max_value=50, value=0)
        prompt = st.sidebar.text_area("Detailed caption prompt", DETAILED_CAPTION_PROMPT, height=140)

    st.sidebar.divider()
    st.sidebar.subheader("System Status")
    status = system_status(device=device, check_model=False)
    loaded = st.session_state.generator_bundle is not None
    css = "status-ok" if loaded else "status-bad"
    text = "Model loaded once and cached" if loaded else "Model will load on first inference"
    st.sidebar.markdown(f'<span class="{css}">{text}</span>', unsafe_allow_html=True)

    if status.get("stage1_config"):
        with st.sidebar.expander("Stage-1 alignment checkpoint"):
            st.json(status["stage1_config"])

    return {
        "device": device,
        "detailed": detailed,
        "beam_size": beam_size,
        "min_new_tokens": min_new_tokens,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "length_penalty": length_penalty,
        "top_k": top_k,
        "prompt": prompt,
    }


def run_inference(image: Image.Image, params: dict):
    generator, _ = get_generator(params["device"])
    prepared = normalize_image(image)
    return generator.generate(
        image=prepared,
        detailed=params["detailed"],
        beam_size=params["beam_size"],
        min_new_tokens=params["min_new_tokens"],
        max_new_tokens=params["max_new_tokens"],
        top_k=params["top_k"],
        temperature=params["temperature"],
        length_penalty=params["length_penalty"],
        prompt=params["prompt"],
    )


def caption_cache_key(image_bytes: bytes, params: dict) -> str:
    relevant_params = (
        params["device"],
        params["detailed"],
        params["beam_size"],
        params["min_new_tokens"],
        params["max_new_tokens"],
        params["top_k"],
        params["temperature"],
        params["length_penalty"],
        params["prompt"],
    )
    digest = hashlib.sha256(image_bytes).hexdigest()
    return repr((digest, relevant_params))


def render_result(result: dict) -> None:
    st.subheader("Detailed caption")
    st.success(result["caption"])

    if result.get("base_caption"):
        st.subheader("Base model caption")
        st.write(result["base_caption"])

    if result.get("satellite_features"):
        with st.expander("Extracted satellite features", expanded=False):
            st.json(result["satellite_features"])

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Inference time", f"{result['inference_time'] * 1000:.0f} ms")
    col_b.metric("Device", result.get("device", "unknown").upper())
    col_c.metric("Precision", result.get("dtype", "default"))


render_header()
params = render_sidebar()

tab_caption, tab_batch, tab_history, tab_status = st.tabs(
    ["Caption", "Batch", "History", "System"]
)

with tab_caption:
    col_upload, col_result = st.columns([1, 1], gap="large")
    with col_upload:
        uploaded = st.file_uploader(
            "Upload a satellite image",
            type=["jpg", "jpeg", "png", "webp"],
        )
        if uploaded:
            uploaded_bytes = uploaded.getvalue()
            image = Image.open(BytesIO(uploaded_bytes))
            preview = normalize_image(image)
            st.image(preview, caption=uploaded.name, use_container_width=True)
            cache_key = caption_cache_key(uploaded_bytes, params)

    with col_result:
        if uploaded:
            result = st.session_state.caption_result_cache.get(cache_key)
            if result is None:
                with st.spinner("Generating caption..."):
                    result = run_inference(image, params)
                st.session_state.caption_result_cache[cache_key] = result
                st.session_state.active_caption_key = cache_key
            if cache_key not in st.session_state.history_keys:
                st.session_state.history.append(
                    {
                        "filename": uploaded.name,
                        "caption": result["caption"],
                        "base_caption": result.get("base_caption", ""),
                        "time_ms": result["inference_time"] * 1000,
                        "model": result["model_id"],
                    }
                )
                st.session_state.history_keys.add(cache_key)

            if result:
                render_result(result)
            else:
                st.info("Upload a satellite image to generate a caption.")
        else:
            st.info("Upload a satellite image to generate a caption.")

with tab_batch:
    files = st.file_uploader(
        "Upload multiple satellite images",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
    )
    if files and st.button("Process batch", type="primary"):
        rows = []
        progress = st.progress(0.0)
        for index, file in enumerate(files):
            image = Image.open(file)
            result = run_inference(image, params)
            rows.append(
                {
                    "Filename": file.name,
                    "Caption": result["caption"],
                    "Base Caption": result.get("base_caption", ""),
                    "Time (ms)": f"{result['inference_time'] * 1000:.0f}",
                    "Model": result["model_id"],
                }
            )
            progress.progress((index + 1) / len(files))

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="vision_gpt_satellite_batch.csv",
            mime="text/csv",
        )

with tab_history:
    if st.session_state.history:
        st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True, hide_index=True)
        if st.button("Clear history"):
            st.session_state.history = []
            st.session_state.history_keys = set()
            st.rerun()
    else:
        st.info("Generated captions will appear here during your session.")

with tab_status:
    status = system_status(device=params["device"], check_model=False)
    st.json(status)
    if st.session_state.generator_bundle is not None:
        st.success("Model loaded once and cached.")
    else:
        st.info("Model has not been loaded yet. It will load on the first caption request.")
