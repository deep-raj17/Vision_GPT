@echo off
cd /d "%~dp0"
echo Installing Vision GPT dependencies...
python -m pip install -r requirements.txt -q
echo Starting Vision GPT on http://localhost:8501
python -m streamlit run app.py --server.headless true --server.port 8501
