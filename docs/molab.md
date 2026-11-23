# Molab: notebook gọi scripts từ GitHub

Notebook: `notebooks/molab_gemma4.ipynb`. Phần xử lý nằm tại
`scripts/molab_benchmark.py`, tải từ GitHub khi chạy cell lấy code.

1. Push notebook và script lên GitHub.
2. Import file `.ipynb` vào runtime Linux NVIDIA GPU, Python >= 3.12.
3. Sửa CONFIG: `valid`, `limit=3` chạy thử; `limit=0` chạy toàn bộ reviewed.
4. Chạy lần lượt: cấu hình → lấy script → cài thư viện → benchmark → kết quả.
5. Nhập HF token khi được hỏi (ô ẩn), hoặc đặt secret `HF_TOKEN` trước.
   Repo code private: `GITHUB_PRIVATE=True`, nhập token hoặc đặt `GITHUB_TOKEN`.
6. Tải ZIP trước khi runtime bị thu hồi. Nếu link không hoạt động, lấy file trong
   thư mục `invoice-results/` bằng trình duyệt file của runtime.

Script tự lấy code cùng commit, tải full HF dataset private và Ollama/model.
Mặc định Gemma 4 31B Q8, chỉ chấm groundtruth reviewed của valid. Không cần DB.
Chỉ mở notebook không chạy model; chạy cell benchmark mới bắt đầu xử lý.

Log hiện trực tiếp, có heartbeat subprocess mỗi 15 giây, traceback và log cuối
khi lỗi. ZIP có predictions, groundtruth đã chọn, metrics, errors, log và phiên
bản code/data/model. Không tự upload kết quả. Token không ghi vào ZIP/source.
Nếu reviewed chưa phủ hết split, điểm chỉ đại diện phần reviewed. Lỗi trích xuất
vẫn tính trong mẫu chấm. Ngưỡng VRAM là kiểm tra sơ bộ; xem server log/offload
trong run manifest khi đo hiệu năng. Ollama dừng sau run, cache giữ khi còn disk.

Notebook tải script ở commit được resolve từ REF (`main` mặc định). Push cập nhật
script rồi chạy lại cell lấy code để dùng bản mới; không cần nhúng code vào notebook.
Bản marimo `.py` cũ vẫn được giữ lại, không cần dùng cho luồng này.
