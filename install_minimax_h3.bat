@echo off
REM Cai ComfyUI + MiniMax H3 (ban nen cua Comfy-Org) len o D:, ~56GB model
REM Doc license truoc: https://huggingface.co/MiniMaxAI/MiniMax-H3
setlocal
set ROOT=D:\ComfyUI

if not exist %ROOT% git clone --depth 1 https://github.com/comfyanonymous/ComfyUI %ROOT%
cd /d %ROOT%
if not exist .venv (
  python -m venv .venv
  .venv\Scripts\python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
  .venv\Scripts\python -m pip install -r requirements.txt
)
.venv\Scripts\python -m pip install -U huggingface_hub

set R=Comfy-Org/MiniMax-H3
.venv\Scripts\hf download %R% diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors --local-dir models\_h3
.venv\Scripts\hf download %R% text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors --local-dir models\_h3
.venv\Scripts\hf download %R% vae/minimax_h3_video_vae_fp16.safetensors vae/minimax_h3_audio_vae_fp32.safetensors --local-dir models\_h3
.venv\Scripts\hf download %R% loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors --local-dir models\_h3

for %%d in (diffusion_models text_encoders vae loras) do (
  if not exist models\%%d mkdir models\%%d
  move /y models\_h3\%%d\*.safetensors models\%%d\ >nul
)
echo.
echo Xong. Chay ComfyUI: D:\ComfyUI\.venv\Scripts\python D:\ComfyUI\main.py --lowvram
pause
