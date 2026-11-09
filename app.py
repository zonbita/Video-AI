"""Video-AI: local image + video generator (Gradio) tuned for 12GB VRAM + 64GB RAM."""
import gc
import os
import time

import gradio as gr
import torch
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

IMAGE_MODELS = {
    "SDXL (nhanh, ~8GB)": "stabilityai/stable-diffusion-xl-base-1.0",
    "FLUX.1-schnell (dep, offload RAM)": "black-forest-labs/FLUX.1-schnell",
}
VIDEO_MODELS = {
    "Wan2.1 T2V 1.3B (~8GB)": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "LTX-Video (nhanh, offload RAM)": "Lightricks/LTX-Video",
}
CUSTOM = "Custom HF repo (vd: MiniMax H3 neu co ban diffusers)"

_cache = {"repo": None, "pipe": None}


def load_pipe(repo: str, offload: str):
    if _cache["repo"] == repo:
        return _cache["pipe"]
    _cache["pipe"] = None
    gc.collect()
    torch.cuda.empty_cache()

    dtype = torch.bfloat16 if ("FLUX" in repo or "Wan" in repo or "LTX" in repo) else torch.float16
    kwargs = {"torch_dtype": dtype}
    if "Wan" in repo:
        from diffusers import AutoencoderKLWan
        kwargs["vae"] = AutoencoderKLWan.from_pretrained(repo, subfolder="vae", torch_dtype=torch.float32)
    pipe = DiffusionPipeline.from_pretrained(repo, **kwargs)

    # 12GB VRAM: offload to the 64GB system RAM
    if offload == "sequential (it VRAM nhat, cham)":
        pipe.enable_sequential_cpu_offload()
    elif offload == "model (can bang)":
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
        try:
            pipe.vae.enable_tiling()
        except Exception:
            pass
    _cache.update(repo=repo, pipe=pipe)
    return pipe


def _repo(choice, table, custom):
    return custom.strip() if choice == CUSTOM else table[choice]


def gen_image(model, custom, prompt, neg, w, h, steps, cfg, seed, offload):
    repo = _repo(model, IMAGE_MODELS, custom)
    pipe = load_pipe(repo, offload)
    g = torch.Generator("cpu").manual_seed(int(seed)) if seed >= 0 else None
    args = dict(prompt=prompt, width=int(w), height=int(h), num_inference_steps=int(steps),
                guidance_scale=float(cfg), generator=g)
    if neg and "FLUX" not in repo:
        args["negative_prompt"] = neg
    img = pipe(**args).images[0]
    path = os.path.join(OUT_DIR, f"img_{int(time.time())}.png")
    img.save(path)
    return img, path


def gen_video(model, custom, prompt, neg, w, h, frames, steps, cfg, fps, seed, offload):
    repo = _repo(model, VIDEO_MODELS, custom)
    pipe = load_pipe(repo, offload)
    g = torch.Generator("cpu").manual_seed(int(seed)) if seed >= 0 else None
    out = pipe(prompt=prompt, negative_prompt=neg or None, width=int(w), height=int(h),
               num_frames=int(frames), num_inference_steps=int(steps),
               guidance_scale=float(cfg), generator=g).frames[0]
    path = os.path.join(OUT_DIR, f"vid_{int(time.time())}.mp4")
    export_to_video(out, path, fps=int(fps))
    return path


def gen_h3(prompt, w, h, seconds, turbo, steps, seed, first, last, progress=gr.Progress()):
    import h3_comfy
    # free VRAM held by diffusers models so ComfyUI gets the whole 12GB
    _cache.update(repo=None, pipe=None)
    gc.collect()
    torch.cuda.empty_cache()
    path, used_seed = h3_comfy.generate(prompt, w, h, seconds, int(seed), turbo, int(steps), OUT_DIR,
                                        first, last, progress)
    return path, f"Seed: {used_seed}"


def gen_h3_ref(prompt, w, h, seconds, turbo, steps, seed, ref_size, r1, r2, r3, progress=gr.Progress()):
    import h3_comfy
    _cache.update(repo=None, pipe=None)
    gc.collect()
    torch.cuda.empty_cache()
    path, used_seed = h3_comfy.generate_ref(prompt, w, h, seconds, int(seed), turbo, int(steps), OUT_DIR,
                                            [r1, r2, r3], ref_size, progress)
    return path, f"Seed: {used_seed}"


OFFLOAD = ["model (can bang)", "sequential (it VRAM nhat, cham)", "none (toan bo tren GPU)"]

with gr.Blocks(title="Video-AI") as demo:
    gr.Markdown("# Video-AI — tao anh & video local (RTX 12GB)")
    with gr.Tab("Anh"):
        with gr.Row():
            with gr.Column():
                im_model = gr.Dropdown(list(IMAGE_MODELS) + [CUSTOM], value=list(IMAGE_MODELS)[0], label="Model")
                im_custom = gr.Textbox(label="HF repo id (khi chon Custom)")
                im_prompt = gr.Textbox(label="Prompt", lines=3)
                im_neg = gr.Textbox(label="Negative prompt")
                with gr.Row():
                    im_w = gr.Slider(512, 1536, 1024, step=64, label="Width")
                    im_h = gr.Slider(512, 1536, 1024, step=64, label="Height")
                with gr.Row():
                    im_steps = gr.Slider(1, 60, 25, step=1, label="Steps (schnell: 4)")
                    im_cfg = gr.Slider(0, 15, 6, step=0.5, label="CFG (schnell: 0)")
                im_seed = gr.Number(-1, label="Seed (-1 = ngau nhien)")
                im_off = gr.Radio(OFFLOAD, value=OFFLOAD[0], label="CPU offload")
                im_btn = gr.Button("Tao anh", variant="primary")
            with gr.Column():
                im_out = gr.Image(label="Ket qua")
                im_file = gr.File(label="File")
        im_btn.click(gen_image, [im_model, im_custom, im_prompt, im_neg, im_w, im_h, im_steps, im_cfg, im_seed, im_off],
                     [im_out, im_file])

    with gr.Tab("Video MiniMax H3"):
        gr.Markdown("Chay qua ComfyUI (`start_comfyui.bat`). Tao video **kem am thanh**, 24 fps.")
        with gr.Row():
            with gr.Column():
                h_prompt = gr.Textbox(label="Prompt (mo ta canh, chuyen dong, am thanh)", lines=5)
                with gr.Row():
                    h_w = gr.Slider(384, 1344, 832, step=32, label="Width")
                    h_h = gr.Slider(384, 1344, 480, step=32, label="Height")
                h_sec = gr.Slider(1, 15, 5, step=0.5, label="Do dai (giay)")
                with gr.Row():
                    h_turbo = gr.Checkbox(True, label="Turbo LoRA (8 buoc, nhanh hon nhieu)")
                    h_steps = gr.Slider(4, 40, 8, step=1, label="Steps (turbo: 8, thuong: 20)")
                h_seed = gr.Number(-1, label="Seed (-1 = ngau nhien)")
                with gr.Row():
                    h_first = gr.Image(type="filepath", label="Khung dau (tuy chon)")
                    h_last = gr.Image(type="filepath", label="Khung cuoi (tuy chon)")
                h_btn = gr.Button("Tao video H3", variant="primary")
            with gr.Column():
                h_out = gr.Video(label="Ket qua")
                h_info = gr.Markdown()
        h_btn.click(gen_h3, [h_prompt, h_w, h_h, h_sec, h_turbo, h_steps, h_seed, h_first, h_last], [h_out, h_info])

    with gr.Tab("H3 Anh tham chieu"):
        gr.Markdown("Giu dung nhan vat/vat the tu anh. Goi anh trong prompt bang **`<Picture 1>`**, "
                    "**`<Picture 2>`**, **`<Picture 3>`** theo dung thu tu o ben duoi.")
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    r_1 = gr.Image(type="filepath", label="<Picture 1>")
                    r_2 = gr.Image(type="filepath", label="<Picture 2> (tuy chon)")
                    r_3 = gr.Image(type="filepath", label="<Picture 3> (tuy chon)")
                r_prompt = gr.Textbox(label="Prompt", lines=5,
                                      placeholder="The swordsman from <Picture 1> stands on a rain-soaked rooftop at night...")
                with gr.Row():
                    r_w = gr.Slider(384, 1344, 832, step=32, label="Width")
                    r_h = gr.Slider(384, 1344, 480, step=32, label="Height")
                r_sec = gr.Slider(1, 15, 5, step=0.5, label="Do dai (giay)")
                with gr.Row():
                    r_turbo = gr.Checkbox(True, label="Turbo LoRA (4 buoc)")
                    r_steps = gr.Slider(4, 40, 4, step=1, label="Steps (turbo: 4, thuong: 20)")
                with gr.Row():
                    r_size = gr.Radio(["match", "max"], value="match",
                                      label="Do net anh tham chieu (max: giong hon nhung cham hon nhieu)")
                    r_seed = gr.Number(-1, label="Seed (-1 = ngau nhien)")
                r_btn = gr.Button("Tao video tu anh tham chieu", variant="primary")
            with gr.Column():
                r_out = gr.Video(label="Ket qua")
                r_info = gr.Markdown()
        r_btn.click(gen_h3_ref, [r_prompt, r_w, r_h, r_sec, r_turbo, r_steps, r_seed, r_size, r_1, r_2, r_3],
                    [r_out, r_info])

    with gr.Tab("Video (Wan / LTX)"):
        with gr.Row():
            with gr.Column():
                v_model = gr.Dropdown(list(VIDEO_MODELS) + [CUSTOM], value=list(VIDEO_MODELS)[0], label="Model")
                v_custom = gr.Textbox(label="HF repo id (khi chon Custom)")
                v_prompt = gr.Textbox(label="Prompt", lines=3)
                v_neg = gr.Textbox(label="Negative prompt", value="blurry, low quality, distorted")
                with gr.Row():
                    v_w = gr.Slider(320, 1280, 832, step=32, label="Width")
                    v_h = gr.Slider(320, 1280, 480, step=32, label="Height")
                with gr.Row():
                    v_frames = gr.Slider(9, 161, 81, step=8, label="Frames (4k+1)")
                    v_fps = gr.Slider(8, 30, 16, step=1, label="FPS")
                with gr.Row():
                    v_steps = gr.Slider(4, 60, 30, step=1, label="Steps")
                    v_cfg = gr.Slider(0, 12, 5, step=0.5, label="CFG")
                v_seed = gr.Number(-1, label="Seed (-1 = ngau nhien)")
                v_off = gr.Radio(OFFLOAD, value=OFFLOAD[0], label="CPU offload")
                v_btn = gr.Button("Tao video", variant="primary")
            with gr.Column():
                v_out = gr.Video(label="Ket qua")
        v_btn.click(gen_video, [v_model, v_custom, v_prompt, v_neg, v_w, v_h, v_frames, v_steps, v_cfg, v_fps, v_seed, v_off],
                    v_out)

if __name__ == "__main__":
    demo.queue().launch(inbrowser=True)
