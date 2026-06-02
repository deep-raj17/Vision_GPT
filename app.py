"""Vision GPT — image captioning web app."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.inference import generate_caption, load_generator, system_status

st.set_page_config(
    page_title="Vision GPT",
    page_icon="👁️",
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


@st.cache_resource
def cached_generator(device: str):
    generator, config = load_generator(device=device)
    return generator, config


def render_header() -> None:
    st.markdown('<div class="main-title">Vision GPT</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Upload an image and generate AI captions using the vision-language pipeline.</div>',
        unsafe_allow_html=True,
    )


def render_sidebar() -> dict:
    st.sidebar.header("Inference Settings")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.sidebar.caption(f"Device: **{device.upper()}**")

    beam_size = st.sidebar.slider("Beam size", 1, 10, 5)
    max_length = st.sidebar.slider("Max caption length", 10, 80, 50)
    temperature = st.sidebar.slider("Temperature", 0.1, 2.0, 1.0, 0.1)
    length_penalty = st.sidebar.slider("Length penalty", 0.0, 1.5, 0.7, 0.05)
    top_k = st.sidebar.number_input("Top-K (0 = beam/greedy)", min_value=0, max_value=50, value=0)

    st.sidebar.divider()
    st.sidebar.subheader("System Status")
    status = system_status(device=device)
    ready = status.get("inference_engine") == "ready"
    css = "status-ok" if ready else "status-bad"
    st.sidebar.markdown(
        f'<span class="{css}">{"Ready" if ready else status.get("inference_engine", "Unknown")}</span>',
        unsafe_allow_html=True,
    )

    if status.get("stage1_config"):
        with st.sidebar.expander("Stage-1 alignment checkpoint"):
            st.json(status["stage1_config"])

    return {
        "device": device,
        "beam_size": beam_size,
        "max_length": max_length,
        "temperature": temperature,
        "length_penalty": length_penalty,
        "top_k": top_k,
        "ready": ready,
    }


render_header()
params = render_sidebar()

tab_caption, tab_batch, tab_history, tab_status = st.tabs(
    ["Caption", "Batch", "History", "System"]
)

with tab_caption:
    if not params["ready"]:
        st.error("Caption model is not ready. Check the System tab for details.")
    else:
        col_upload, col_result = st.columns([1, 1], gap="large")
        with col_upload:
            uploaded = st.file_uploader(
                "Upload an image",
                type=["jpg", "jpeg", "png", "webp"],
            )
            if uploaded:
                image = Image.open(uploaded).convert("RGB")
                st.image(image, caption=uploaded.name, use_container_width=True)

        with col_result:
            if uploaded:
                with st.spinner("Generating caption..."):
                    generator, _ = cached_generator(params["device"])
                    result = generator.generate(
                        image=image,
                        beam_size=params["beam_size"],
                        max_length=params["max_length"],
                        top_k=params["top_k"],
                        temperature=params["temperature"],
                        length_penalty=params["length_penalty"],
                    )

                st.success(result["caption"])
                st.metric("Confidence", f"{result['confidence'] * 100:.2f}%")
                st.metric("Inference time", f"{result['inference_time'] * 1000:.0f} ms")

                if len(result.get("all_candidates", [])) > 1:
                    st.subheader("Alternative captions")
                    for idx, candidate in enumerate(result["all_candidates"][:5], start=1):
                        st.write(f"{idx}. {candidate['caption']}")

                st.session_state.history.append(
                    {
                        "filename": uploaded.name,
                        "caption": result["caption"],
                        "confidence": result["confidence"],
                        "time_ms": result["inference_time"] * 1000,
                    }
                )
            else:
                st.info("Upload an image to generate a caption.")

with tab_batch:
    if not params["ready"]:
        st.error("Caption model is not ready.")
    else:
        files = st.file_uploader(
            "Upload multiple images",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
        )
        if files and st.button("Process batch", type="primary"):
            generator, _ = cached_generator(params["device"])
            rows = []
            progress = st.progress(0.0)
            for index, file in enumerate(files):
                image = Image.open(file).convert("RGB")
                result = generator.generate(
                    image=image,
                    beam_size=params["beam_size"],
                    max_length=params["max_length"],
                    top_k=params["top_k"],
                    temperature=params["temperature"],
                    length_penalty=params["length_penalty"],
                )
                rows.append(
                    {
                        "Filename": file.name,
                        "Caption": result["caption"],
                        "Confidence": f"{result['confidence'] * 100:.1f}%",
                        "Time (ms)": f"{result['inference_time'] * 1000:.0f}",
                    }
                )
                progress.progress((index + 1) / len(files))

            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.download_button(
                "Download CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="vision_gpt_batch.csv",
                mime="text/csv",
            )

with tab_history:
    if st.session_state.history:
        st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True, hide_index=True)
        if st.button("Clear history"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("Generated captions will appear here during your session.")

with tab_status:
    status = system_status(device=params["device"])
    st.json(status)

    st.markdown(
        """
        **Components**
        - **Caption model** — BLIP (`Salesforce/blip-image-captioning-base`) via HuggingFace
        - **Stage-1 alignment** — CLIP vision encoder + projection weights for GPT-OSS (stored locally)
        - **COCO annotations** — training data artifacts in `vision_gpt_backend/Data`
        """
    )
