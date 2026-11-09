# Ghi chú: MiniMax H3 trên VGA 12GB (09/11/2026)

## Số liệu đo thực tế (832×480, 24fps, LoRA turbo 8 bước)

| Độ dài | Frame | Tốc độ | Tổng thời gian |
|---|---|---|---|
| ~5s | 124 | ~57 s/bước | **~10 phút** |
| ~15s | 362 | ~246 s/bước (chậm ~4x) | **~40 phút** |

## Vì sao video dài chậm hơn nhiều
- Model MiniMax H3 (int8) khoảng **20 GB**, lớn hơn VRAM 12 GB, nên phần lớn model nằm trong RAM (`--lowvram`, "dynamic VRAM loading").
- GPU chỉ tính toán trên VRAM, nên phải liên tục chép model từ RAM sang qua PCIe. Đường này chậm hơn VRAM khoảng 15–30 lần.
- Phần bộ nhớ cho các frame đang tạo **bắt buộc nằm trong VRAM**, vì model xử lý mọi frame cùng lúc. Càng nhiều frame thì càng ít chỗ cho model, nên phải chép nhiều hơn và chạy chậm hơn.
- GPU báo 100% không có nghĩa là đang chạy nhanh: nhiều khi nó đang chờ dữ liệu chép sang.
- Chạy hẳn bằng CPU thì chậm hơn GPU khoảng 50–100 lần, không thực tế.

## Độ dài nên dùng với 12 GB
| Độ dài | Frame | Thời gian ước tính |
|---|---|---|
| **5s (nên dùng)** | ~121 | ~10 phút (đã đo) |
| 7–8s | ~169–193 | ~15–20 phút (ước tính) |
| 10s | ~241 | ~25–30 phút (ước tính) |
| 15s | ~362 | ~40 phút (đã đo) |

Với video dài, nên **chia thành các đoạn 5 giây** (mỗi đoạn 1 cảnh) rồi ghép lại. Cách này nhanh hơn và mỗi cảnh thường đẹp hơn.
Số frame nên có dạng 8n+1 (121, 169, 193, 241…). Đây là quy ước của nhiều model video, chưa kiểm tra với MiniMax H3.

## VRAM cần cho 15s (362 frame), ước tính
| VRAM | Thời gian ước tính |
|---|---|
| 12 GB | ~40 phút (đã đo) |
| 16 GB | ~20–25 phút |
| 24 GB (3090/4090) | ~10–12 phút (mức tối thiểu để gần như không phải chép từ RAM) |
| 32 GB (5090) | ~6–8 phút (chứa hết model và frame) |

Cơ sở ước tính: model khoảng 20 GB cộng phần frame khoảng 4–8 GB, tổng **khoảng 24–28 GB**.

## Lưu ý khác
- **Không ngắt được giữa chừng để lấy video dở**: Interrupt sẽ hủy cả job và không lưu gì.
- Xem tiến độ bằng log ComfyUI: `http://127.0.0.1:8188/internal/logs/raw` (dòng `x/8 [..., s/it]`).
- Mỗi chương trình chạy bằng `.venv` hiện thành 2 tiến trình Python (launcher `.venv` là cha, `C:\Python312` là con). **Đó không phải bản trùng, đừng tắt.**
