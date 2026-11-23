# Ollama local trên Windows

Luồng: upload ảnh → model → kiểm tra JSON → xem PO/items/totals → bấm lưu DB.
Ollama chạy như dịch vụ riêng, nên Python không cần PyTorch/BitsAndBytes cho
backend này. Một lần upload là một hóa đơn, có thể nhiều trang.

## Chuẩn bị và chạy

Cần Python 3.10+, [Ollama Windows](https://ollama.com/download/windows) và
PostgreSQL. Qwen3-VL cần Ollama 0.12.7 trở lên. Nếu dùng Docker cho DB, mở
Docker Desktop trước. Script không tự cài Ollama hoặc Docker Desktop.

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_local.ps1 -Install -PullModel -StartDatabase -InitializeDatabase
```

Mở http://127.0.0.1:7860; giữ terminal chạy. Ctrl+C dừng app. Lệnh trên có thể
chạy lại: giữ cấu hình đang có, chỉ tạo phần còn thiếu. `.env` cũ chưa khai báo
backend sẽ được bổ sung Ollama. Nếu đã chọn Transformers, script giữ lựa chọn đó.

Những lần sau:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_local.ps1
```

Thêm `-StartDatabase` nếu container DB đang dừng. Kiểm tra mà không mở UI:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_local.ps1 -CheckOnly
```

Nếu có PostgreSQL sẵn, đặt INVOICE_DATABASE_URL rồi bỏ `-StartDatabase`.
Dùng `-EnvFile PATH` và `-Port 7861` khi cần cấu hình/cổng khác.

## Cài và chạy thủ công

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements/local.txt -e .
Copy-Item configs/ollama.env.example .env
ollama pull qwen3-vl:2b
.venv/Scripts/python.exe -m invoice_ocr ollama doctor
.venv/Scripts/python.exe -m invoice_ocr ui
```

Chỉ copy mẫu khi chưa có `.env`; giữ connection string đang sử dụng nếu đã có.

## Tham số chính trong .env

```dotenv
INVOICE_BACKEND=ollama
INVOICE_OLLAMA_URL=http://127.0.0.1:11434
INVOICE_OLLAMA_MODEL=qwen3-vl:2b
INVOICE_OLLAMA_DEVICE=cpu
INVOICE_OLLAMA_TIMEOUT=600
INVOICE_OLLAMA_NUM_CTX=8192
INVOICE_OLLAMA_NUM_THREADS=
INVOICE_OLLAMA_KEEP_ALIVE=5m
INVOICE_MAX_EDGE=1600
INVOICE_MAX_NEW_TOKENS=4096
```

CPU gửi `num_gpu=0`; auto để Ollama phân bổ phần cứng. Đổi tên model và chạy
lại với `-PullModel` để tải model khác. Model phải hỗ trợ ảnh. CPU có thể mất
nhiều thời gian; tăng TIMEOUT nếu cần. Context/output lớn hơn cũng cần thêm RAM.

## Upload và kết quả

Nhận JPG/JPEG/PNG/WEBP/TIFF một frame, tối đa 20 MB/file. Thứ tự trong danh sách
upload là thứ tự trang. Không nhận PDF hoặc tự bỏ bớt trang. Ảnh gốc giữ nguyên;
MAX_EDGE chỉ giới hạn bản đưa vào model.

```text
<INVOICE_OUTPUT_DIR>/uploads/<uuid>/
  images/<uuid>/page_0001.png, page_0002.jpg, ...
  raw/<uuid>.json
```

Kết quả gồm metadata model/prompt/ảnh, raw_text và extraction. JSON tải xuống
giữ Decimal chính xác. Model lỗi/JSON sai schema vẫn có file lỗi, không được
lưu như hóa đơn hợp lệ. DB lỗi không làm mất JSON; có thể lưu lại sau khi sửa DB.
Các file upload/prediction không tự xóa.

App bind 127.0.0.1, không tạo link public. Không có DB vẫn trích xuất/tải JSON
được, nhưng nút lưu bị tắt. Xem [DB](database.md), [GPU/SSH](gpu.md),
[benchmark](benchmark.md) cho các bước tiếp theo.
