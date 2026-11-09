# Video-AI

App Gradio tạo **ảnh** và **video** chạy local, tối ưu cho máy **RTX 3060 12GB VRAM + 64GB RAM**.

## Chạy app

Mỗi lần mở máy, chạy theo thứ tự:

1. `start_comfyui.bat` — khởi động ComfyUI (backend cho MiniMax H3) tại http://127.0.0.1:8188, chế độ `--lowvram`.
2. `run.bat` — mở app tại http://127.0.0.1:7860.

Lần đầu `run.bat` tự tạo `.venv` và cài thư viện. Kết quả lưu ở thư mục `outputs/`.

> Nếu trình duyệt đang mở trang app cũ sau khi khởi động lại app, nhấn **F5**. Trang cũ mất kết nối và sẽ không bao giờ trả kết quả.

## Các tab trong app

| Tab | Model | Ghi chú |
|---|---|---|
| Anh | SDXL, FLUX.1-schnell | SDXL: 25 steps, CFG 6. FLUX schnell: 4 steps, CFG 0, cần đăng nhập Hugging Face |
| Video MiniMax H3 | MiniMax H3 FL2VA (qua ComfyUI) | Text-to-video có âm thanh, tuỳ chọn khung đầu/khung cuối |
| H3 Anh tham chieu | MiniMax H3 Ref2VA (qua ComfyUI) | Giữ nhân vật từ ảnh, gọi ảnh trong prompt bằng `<Picture 1>`, `<Picture 2>`, `<Picture 3>` |
| Video (Wan / LTX) | Wan2.1 T2V 1.3B, LTX-Video (diffusers) | Model nhỏ, ô Custom cho HF repo khác |

## MiniMax H3

- Weights mở từ khoảng 03/08/2026, repo gốc `MiniMaxAI/MiniMax-H3` (~143GB bf16, không vừa ổ).
- Dùng bản nén của **`Comfy-Org/MiniMax-H3`**, chạy qua ComfyUI ở `D:\ComfyUI`.
- Chọn **int8** vì RTX 3060 (Ampere) không chạy fp8 / nvfp4 trực tiếp.

### File model (trong `D:\ComfyUI\models\`)

| Thư mục | File | Dung lượng |
|---|---|---|
| diffusion_models | `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | 21GB |
| diffusion_models | `minimax_h3_ref2va_pruned_int8_convrot.safetensors` | 21GB |
| text_encoders | `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` | 27.1GB |
| vae | `minimax_h3_video_vae_fp16.safetensors` | 5.2GB |
| vae | `minimax_h3_audio_vae_fp32.safetensors` | 0.6GB |
| loras | `minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors` | 2GB |
| loras | `minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors` | 2GB |

Cài lại từ đầu: `install_minimax_h3.bat` (cài ComfyUI + tải bộ FL2VA). Bộ Ref2VA tải riêng bằng `hf download`.

### Giới hạn

- **Độ phân giải:** khung gốc cạnh ngắn 768px, tối đa **1344×768**. Không xuất thẳng Full HD hay 2K.
- **2K "Regenerate"** chỉ có qua API trả phí của MiniMax, không có trong bản weights mở.
- **Độ dài:** 24 fps, số frame theo dạng 17k+5 (124 frame ≈ 5 giây). Khoảng huấn luyện 124–362 frame.
- **License:** MiniMax H3 Community License. Bản weights mở **chưa được phép dùng ở EU, Anh, Hàn Quốc, Mỹ** (theo `docs/QA-about-License.md` trong repo). Đọc license trước khi dùng thương mại.

### Tốc độ đo được trên RTX 3060

Video thử 5 giây, 832×480, turbo 8 bước:

- Nạp model lần đầu: ~3 phút.
- Mỗi bước: ~56 giây.
- **Tổng: 14 phút 19 giây** (gồm cả nạp model).

Video dài hơn chậm hơn nhiều hơn tỉ lệ thuận (vd. 12 giây ước 25–45 phút, chưa đo). Nên thử 5–6 giây trước.

### Cách H3 tạo video

H3 **không** vẽ từng tấm rồi ghép. Nó khử nhiễu **cả khối video và âm thanh cùng lúc** (8 bước turbo), rồi VAE giải nén ra 124 frame và âm thanh, cuối cùng ghép thành MP4. Hệ quả:

- Không có bản xem trước, phải chờ đến cuối.
- Không chia một video cho 2 GPU / 2 máy được (qua mạng LAN còn chậm hơn 1 máy).

### Dùng 2 máy

- **Mỗi máy một cảnh:** chia prompt theo mốc thời gian, mỗi máy tạo một phần, rồi ghép lại.
- **Cùng prompt, khác seed:** ra 2 phiên bản để chọn. Cùng seed + cùng prompt thì ra gần như cùng một video.
- **Chia việc:** một máy tạo video, máy kia upscale.
- Cùng seed **không** giữ nhân vật giống nhau giữa các cảnh khác prompt. Muốn giữ nhân vật, dùng **khung đầu** hoặc **Ref2VA**.

### Mẹo Ref2VA

- Prompt rất nhạy với cách viết: gọi đúng thẻ `<Picture N>` theo thứ tự ảnh, nói rõ ảnh nào điều khiển phần nào của cảnh.
- `ref_image_size`: `match` nhanh; `max` giống nhân vật hơn nhưng chậm hơn nhiều lần.
- Scheduler `beta` (mẫu chính thức khuyên dùng thay `simple` cho prompt nhiều tham chiếu).

## Full HD / 2K

### Model local hỗ trợ Full HD

| Model | Full HD | 12GB VRAM |
|---|---|---|
| LTX-2.5 | Tạo thẳng 1920×1088, có âm thanh | Chưa chắc (nguồn nói 12GB, nguồn khác 24GB) |
| LTX-2.3 | Tạo thẳng 1920×1088 | Được (tối thiểu 12GB theo một nguồn), chậm |
| LTX-2 | Mặc định 720p, đẩy được lên 1920×1088 | Chỉ nên thử nếu GPU mạnh |
| HunyuanVideo 1.5 | 720p + model upsampler lên 1080p | Có khả năng chạy được |
| Wan 2.2 | ~720p | Không đạt Full HD |
| MiniMax H3 | Tối đa 1344×768 | Không đạt Full HD |

LTX: miễn phí cho công ty doanh thu dưới ~10 triệu USD/năm, đọc license gốc nếu dùng thương mại.

### Upscale video 768p lên 2K

- **SeedVR2 3B int8 (khuyên dùng):** ComfyUI có sẵn node và mẫu `utility_seedvr2_3b_int8_upscale_video`. Model từ `Comfy-Org/SeedVR2`:
  - `seedvr2_3b_int8_convrot.safetensors` (3.22GB) → `models/diffusion_models`
  - `seedvr2_ema_vae_fp16.safetensors` (478MB) → `models/vae`
  - Xử lý nhiều frame cùng lúc nên không nhấp nháy. Thiếu VRAM thì dùng `SeedVR2TemporalChunk`. Chưa thử trên máy này.
- **Real-ESRGAN:** nhanh, file ~60MB, chạy bằng node `UpscaleModelLoader` + `ImageUpscaleWithModel`. Phóng từng tấm nên dễ nhấp nháy.
- **Không nên** cho H3 tạo thẳng 2560×1440: ngoài khoảng huấn luyện, dễ méo/lặp và gần như chắc hết VRAM.

## Ghi chú cài đặt

- Torch phải cài bản CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu128` (cài từ PyPI thường sẽ ra bản CPU). `run.bat` đã làm sẵn.
- Tải từ Hugging Face không đăng nhập bị giới hạn tốc độ. Đặt biến môi trường `HF_TOKEN` để tải nhanh hơn.
- Dung lượng ổ D sau khi cài đủ FL2VA + Ref2VA: còn trống khoảng 24GB.

## Cấu trúc file

| File | Vai trò |
|---|---|
| `app.py` | Giao diện Gradio, các tab ảnh/video |
| `h3_comfy.py` | Dựng đồ thị ComfyUI cho H3 (theo mẫu chính thức `video_minimax_h3_t2v` / `video_minimax_h3_r2v`), gửi việc và lấy video về |
| `requirements.txt` | Thư viện Python (torch CUDA cài riêng) |
| `run.bat` | Tạo venv, cài thư viện, mở app |
| `start_comfyui.bat` | Khởi động ComfyUI cho H3 |
| `install_minimax_h3.bat` | Cài ComfyUI + tải model H3 FL2VA |
