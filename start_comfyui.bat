@echo off
REM ComfyUI backend cho MiniMax H3 (12GB VRAM -> --lowvram, offload sang RAM)
cd /d D:\ComfyUI
.venv\Scripts\python main.py --lowvram --port 8188
