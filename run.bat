@echo off
cd /d %~dp0
if not exist .venv (
  python -m venv .venv
  .venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
  .venv\Scripts\python -m pip install -r requirements.txt
)
.venv\Scripts\python app.py
