# Satellite Image Detailed Captioning and Scene Understanding

Vision GPT is a Streamlit app for grounded satellite-image captioning using HuggingFace vision-language models. The goal is no longer generic COCO-style captioning; the app now produces SAT-VISION-style remote-sensing analysis based only on visible image evidence.

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Or double-click `run.bat` on Windows.

Open **http://localhost:8501** and upload a satellite image.

## Model Strategy

The default model is:

```text
Salesforce/blip-image-captioning-base
```

BLIP is not an instruction-following model, so the app does not send it a long request like "describe this satellite image in detail." For BLIP, the backend uses only this short conditional prompt:

```text
a satellite image of
```

The backend then cleans the base BLIP caption, extracts conservative image cues, and expands only the supported evidence into a 200-300 word remote-sensing paragraph. It avoids location names, historical claims, disaster claims, industrial labels, agriculture, or named water bodies unless those facts are visually supported.

BLIP generation settings:

```python
model.generate(
    **inputs,
    max_new_tokens=35,
    min_new_tokens=8,
    num_beams=3,
    repetition_penalty=1.4,
    length_penalty=1.0,
    no_repeat_ngram_size=3,
    early_stopping=True,
)
```

Streamlit loading is optimized for repeated use:

- The model and processor are cached with `st.cache_resource`.
- The inference object is stored in `st.session_state`.
- Device selection prefers CUDA, then MPS, then CPU.
- CUDA inference uses `torch.float16`.
- Prediction runs under `torch.inference_mode()`.
- Large uploads are resized before inference while preserving aspect ratio.

You can try stronger models by setting `CAPTION_MODEL_ID` in `.env`:

```text
CAPTION_MODEL_ID=Salesforce/blip2-opt-2.7b
CAPTION_MODEL_ID=llava-hf/llava-1.5-7b-hf
```

Large BLIP, BLIP-2, and LLaVA models need much more RAM/VRAM than BLIP base. The recommended default is BLIP base for a fast short caption, then the custom satellite expansion stage for the 200-300 word final analyst caption.

For true expert captions, the intended long-term path is structured remote-sensing interpretation:

```text
Satellite image
-> Remote-sensing vision encoder
-> Object / scene detection
-> Buildings, roads, water, vegetation, bare land, coastline, rivers/canals
-> Scene graph
-> Large language model
-> Detailed faithful caption
```

The current BLIP path is a lightweight prototype of that idea: BLIP produces a short base caption, simple image cues provide structured evidence, and a strict composer turns only supported facts into a 200-300 word SAT-VISION analyst caption ending with an overall confidence label. It does not infer city names, country names, historical sites, disaster events, industrial activity, agriculture, or named rivers from pixels alone. For production quality, replace the lightweight cues with trained land-cover and object detectors, then fine-tune or prompt an LLM with the resulting scene graph.

## Dataset Integration

The project includes download scripts for building a satellite-caption dataset:

```bash
python scripts/download_copernicus_sentinel2.py
python scripts/download_bhoonidhi.py
python scripts/download_nasa_lance.py
python scripts/download_usgs_tnm.py
```

Copy `.env.example` to `.env` and fill in credentials before downloading. Do not commit real credentials.

Copernicus downloads Sentinel-2 L2A optical imagery into:

```text
satellite_caption_dataset/raw_images/copernicus/
```

Bhoonidhi downloads selected NRSC products into:

```text
satellite_caption_dataset/raw_images/bhoonidhi/
```

Bhoonidhi uses `/auth/token`, `/data/search`, and `/download`. API access may require your static public IPv4 address to be enabled by Bhoonidhi/NRSC before requests work. The downloader filters for `Online = "Y"` because only online products can be downloaded through the API.

Change one `.env` value to switch datasets:

```text
BHOONIDHI_COLLECTION=Sentinel-1A_SAR-IW_GRD
```

Other collection examples from the API specification:

```text
Sentinel-1A_SAR-IW_SLC
EOS-04_SAR-MRS_L2A
EOS-04_SAR-MRS_L2B
EOS-06_OCM-LAC_L1C
EOS-06_OCM-GAC_L1C
ResourceSat-2_LISS3_L2
ResourceSat-2A_LISS3_L2
ResourceSat-2_AWIFS_L2
CartoSat-1_PAN_CartoDEM_30m
NISAR_SSAR-Beta_RSLC
NISAR_SSAR-Beta_GSLC
NISAR_SSAR-Beta_GCOV
```

NASA LANCE downloads Near Real-Time MODIS/VIIRS products into:

```text
satellite_caption_dataset/raw_images/nasa_lance/
```

The default NASA product is `MOD021KM` from:

```text
https://nrt3.modaps.eosdis.nasa.gov/archive/allData/61/MOD021KM/Recent/
```

Set `NASA_EARTHDATA_TOKEN` in `.env`. You can switch to VIIRS by changing:

```text
NASA_LANCE_COLLECTION_VERSION=5200
NASA_LANCE_PRODUCT=VNP02IMG
```

USGS TNM downloads selected National Map datasets for your area of interest. It uses direct `downloadURL` links from:

```text
https://tnmaccess.nationalmap.gov/api/v1/products
```

The default query downloads a small, captioning-oriented set of National Map products for a Washington DC bounding box and writes a manifest CSV:

```text
USGS_TNM_DATASETS=Digital Elevation Model (DEM) 1 meter,US Topo,National Hydrography Dataset (NHD),National Transportation Dataset (NTD)
USGS_TNM_BBOX=-77.12,38.80,-76.90,39.00
USGS_TNM_MAX_PER_DATASET=50
```

Common dataset names to try:

```text
Digital Elevation Model (DEM) 1 meter
Digital Elevation Model (DEM) 1/3 arc-second
Digital Elevation Model (DEM) 1 arc-second
National Hydrography Dataset (NHD)
Watershed Boundary Dataset (WBD)
National Transportation Dataset (NTD)
National Structures Dataset
US Topo
Historical Topographic Map Collection
National Boundaries Dataset
NAIP Plus
```

You cannot download all U.S. National Map data at once; it is huge. This script downloads all matching products for the datasets and bounding box you select. `NAIP Plus` is the most image-like option, but availability may vary and USGS commonly recommends EarthExplorer for bulk NAIP workflows.

## Fine-Tuning Direction

For real remote-sensing captions, replace COCO training artifacts with satellite-caption datasets such as:

- RSICD
- UCM-Captions
- Sydney-Captions
- NWPU-Captions

Use the downloaded Copernicus, Bhoonidhi, NASA LANCE, or USGS TNM data as raw inputs, then pair images with human captions or curated labels before fine-tuning.

## Project Structure

```text
Vision_GPT/
|-- app.py
|-- run.bat
|-- requirements.txt
|-- .env.example
|-- backend/
|   |-- config.py
|   `-- inference.py
|-- scripts/
|   |-- download_copernicus_sentinel2.py
|   |-- download_bhoonidhi.py
|   |-- download_nasa_lance.py
|   `-- download_usgs_tnm.py
|-- satellite_caption_dataset/
|   `-- raw_images/
`-- vision_gpt_backend/
    |-- Data/
    `-- Output/
```

The local stage-1 alignment checkpoint is preserved for future GPT-OSS integration.
