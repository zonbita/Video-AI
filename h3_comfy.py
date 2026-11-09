"""MiniMax H3 text/image-to-video via a local ComfyUI server (graph mirrors the official
video_minimax_h3_t2v template)."""
import json
import os
import random
import time
import urllib.parse
import urllib.request
import uuid

COMFY = os.environ.get("COMFY_URL", "http://127.0.0.1:8188")

UNET = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
CLIP = "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
VAE_V = "minimax_h3_video_vae_fp16.safetensors"
VAE_A = "minimax_h3_audio_vae_fp32.safetensors"
TURBO = "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"


def _req(path, data=None, headers=None):
    req = urllib.request.Request(COMFY + path, data=data, headers=headers or {})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def is_up():
    try:
        _req("/system_stats")
        return True
    except Exception:
        return False


def frames_for(seconds: float) -> int:
    # same formula as the template: 24 fps, snapped up to the 17k+5 grid
    n = max(5, round(seconds * 24))
    return n + (5 - n % 17) % 17


def upload_image(path: str) -> str:
    boundary = uuid.uuid4().hex
    name = os.path.basename(path)
    with open(path, "rb") as f:
        body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\n"
                f"Content-Type: application/octet-stream\r\n\r\n").encode() + f.read() + \
               f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n--{boundary}--\r\n".encode()
    res = json.loads(_req("/upload/image", body, {"Content-Type": f"multipart/form-data; boundary={boundary}"}))
    return res["name"]


def build_graph(prompt, width, height, seconds, seed, turbo, steps, first_frame=None, last_frame=None):
    g = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": UNET, "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": CLIP, "type": "minimax", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": VAE_V}},
        "avae": {"class_type": "VAELoader", "inputs": {"vae_name": VAE_A}},
        "cond": {"class_type": "MiniMaxH3ImageToVideo", "inputs": {
            "clip": ["clip", 0], "vae": ["vae", 0], "prompt": prompt,
            "width": int(width), "height": int(height), "length": frames_for(seconds)}},
        "noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": int(seed)}},
        "sampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "sched": {"class_type": "BasicScheduler", "inputs": {
            "model": ["model", 0], "scheduler": "simple", "steps": int(steps), "denoise": 1.0}},
        "guider": {"class_type": "BasicGuider", "inputs": {"model": ["model", 0], "conditioning": ["cond", 0]}},
        "sample": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["sampler", 0],
            "sigmas": ["sched", 0], "latent_image": ["cond", 1]}},
        "dec": {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}},
        "adec": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["sample", 0], "vae": ["avae", 0]}},
        "mk": {"class_type": "CreateVideo", "inputs": {"images": ["dec", 0], "audio": ["adec", 0], "fps": 24}},
        "save": {"class_type": "SaveVideo", "inputs": {
            "video": ["mk", 0], "filename_prefix": "video/MiniMax_H3", "format": "auto", "codec": "auto"}},
    }
    if turbo:
        g["model"] = {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": ["unet", 0], "lora_name": TURBO, "strength_model": 1.0}}
    else:
        g["sched"]["inputs"]["model"] = ["unet", 0]
        g["guider"]["inputs"]["model"] = ["unet", 0]
    for key, img in (("first_frame", first_frame), ("last_frame", last_frame)):
        if img:
            node = f"load_{key}"
            g[node] = {"class_type": "LoadImage", "inputs": {"image": upload_image(img)}}
            g["cond"]["inputs"][key] = [node, 0]
    return g


REF_UNET = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
REF_TURBO = "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"


def build_ref_graph(prompt, width, height, seconds, seed, turbo, steps, ref_images, ref_size="match"):
    """Mirrors the official video_minimax_h3_r2v template (reference-to-video)."""
    g = build_graph(prompt, width, height, seconds, seed, turbo, steps)
    g["unet"]["inputs"]["unet_name"] = REF_UNET
    if turbo:
        g["model"]["inputs"]["lora_name"] = REF_TURBO
    # template note: beta/normal scheduler beats simple for reference-heavy prompts
    g["sched"]["inputs"]["scheduler"] = "beta"
    cond = {"clip": ["clip", 0], "vae": ["vae", 0], "audio_vae": ["avae", 0], "prompt": prompt,
            "width": int(width), "height": int(height), "length": frames_for(seconds), "ref_image_size": ref_size}
    for i, img in enumerate(ref_images):
        g[f"ref{i}"] = {"class_type": "LoadImage", "inputs": {"image": upload_image(img)}}
        cond[f"ref_images.ref_image_{i}"] = [f"ref{i}", 0]
    g["cond"] = {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": cond}
    return g


def generate_ref(prompt, width, height, seconds, seed, turbo, steps, out_dir, ref_images,
                 ref_size="match", progress=None):
    refs = [r for r in ref_images if r]
    if not refs:
        raise RuntimeError("Can it nhat 1 anh tham chieu.")
    return generate(prompt, width, height, seconds, seed, turbo, steps, out_dir, progress=progress,
                    _graph=lambda s: build_ref_graph(prompt, width, height, seconds, s, turbo, steps, refs, ref_size))


def generate(prompt, width, height, seconds, seed, turbo, steps, out_dir,
             first_frame=None, last_frame=None, progress=None, _graph=None):
    if not is_up():
        raise RuntimeError(f"ComfyUI chua chay tai {COMFY}. Chay start_comfyui.bat truoc.")
    if seed < 0:
        seed = random.randint(0, 2**48)
    graph = _graph(seed) if _graph else build_graph(prompt, width, height, seconds, seed, turbo, steps,
                                                     first_frame, last_frame)
    body = json.dumps({"prompt": graph, "client_id": uuid.uuid4().hex}).encode()
    res = json.loads(_req("/prompt", body, {"Content-Type": "application/json"}))
    if res.get("node_errors"):
        raise RuntimeError(f"ComfyUI bao loi graph: {res['node_errors']}")
    pid = res["prompt_id"]
    t0 = time.time()
    while True:
        hist = json.loads(_req(f"/history/{pid}"))
        if pid in hist:
            h = hist[pid]
            st = h.get("status", {})
            if st.get("status_str") == "error":
                msgs = [m for m in st.get("messages", []) if m[0] == "execution_error"]
                raise RuntimeError(f"ComfyUI loi: {msgs[-1][1].get('exception_message') if msgs else st}")
            for out in h.get("outputs", {}).values():
                for f in out.get("images", []) + out.get("videos", []) + out.get("gifs", []):
                    q = urllib.parse.urlencode({"filename": f["filename"], "subfolder": f.get("subfolder", ""),
                                                "type": f.get("type", "output")})
                    dst = os.path.join(out_dir, f"h3_{int(time.time())}_{f['filename']}")
                    with open(dst, "wb") as w:
                        w.write(_req(f"/view?{q}"))
                    return dst, seed
            if st.get("completed"):
                raise RuntimeError("ComfyUI chay xong nhung khong tra ve video.")
        if progress:
            progress(min(0.99, (time.time() - t0) / 900), desc=f"MiniMax H3 dang tao... {int(time.time() - t0)}s")
        time.sleep(3)
