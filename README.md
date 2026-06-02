# Vision GPT

AI image captioning app powered by BLIP (HuggingFace).

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Or double-click `run.bat` on Windows.

Open **http://localhost:8501** and upload an image.

## Project structure

```
vision_gpt_frontend/
├── app.py                  # Streamlit UI
├── run.bat                 # Windows launcher
├── requirements.txt
├── backend/
│   ├── config.py           # Paths and model settings
│   └── inference.py        # BLIP caption engine
└── vision_gpt_backend/
    ├── Data/               # COCO training annotations
    └── Output/             # Stage-1 alignment checkpoint
```

## Model

Captions are generated with **Salesforce/blip-image-captioning-base** — a pretrained vision-language model that produces natural English descriptions (no `<UNK>` tokens).

The local stage-1 alignment checkpoint (CLIP + projection for GPT-OSS) is preserved for future LLM integration.
