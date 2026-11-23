# Invoice OCR

Upload các ảnh của **một hóa đơn**, trích xuất company/bill/ship, số hóa đơn,
ngày, PO, thông tin xe, items và totals; kiểm tra JSON rồi lưu PostgreSQL.
Một hóa đơn nhiều trang vẫn trả một JSON `invoice-v3`. Chưa nhận PDF trực tiếp
ở bước inference; PDF được render trong bước chuẩn bị dữ liệu.

## Chạy local bằng Ollama, không cần GPU

Cài Python 3.10+, [Ollama](https://ollama.com/download/windows), mở Ollama và
Docker Desktop. Chạy trong thư mục dự án:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_local.ps1 -Install -PullModel -StartDatabase -InitializeDatabase
```

Mở **http://127.0.0.1:7860**. Script tạo `.venv`/`.env` nếu thiếu, cài package,
tải `qwen3-vl:2b`, bật PostgreSQL và tạo bảng. Nếu chưa có cấu hình DB, script
sinh mật khẩu local trong `.env`; không in mật khẩu ra log.

Các lần sau, khi Ollama và DB đang hoạt động:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_local.ps1
```

## Các lệnh chính

Sau khi kích hoạt `.venv` và cài editable package bằng `python -m pip install -e .`:

```text
invoice-ocr ui
invoice-ocr ollama doctor
invoice-ocr extract --input ... --output ...
invoice-ocr evaluate --groundtruth ... --predictions ... --output ...
invoice-ocr db init
invoice-ocr db load --input ...
invoice-ocr gpu setup|doctor|download
invoice-ocr label prepare|review|generate|verify|export
```

`|` biểu thị các lựa chọn, không phải lệnh shell cần chạy nguyên dòng.
Mọi lệnh cũng chạy được bằng `python -m invoice_ocr ...`. Dùng `--help` để xem
tham số của từng lệnh, ví dụ `invoice-ocr label generate --help`.

## Thư mục

| Thư mục | Trách nhiệm |
| --- | --- |
| `app/` | Giao diện và điều phối upload/lưu DB |
| `src/invoice_ocr/` | Package chung: model, schema, lưu trữ, labeling, evaluation, CLI |
| `scripts/` | Chuẩn bị môi trường, khởi động local và chia dataset |
| `configs/` | Cấu hình mẫu cho Ollama, Transformers và labeling |
| `database/` | Schema PostgreSQL |
| `deploy/` | Docker Compose |
| `requirements/` | Bộ thư viện base/local/gpu/labeling |
| `notebooks/` | Kaggle/Colab GPU, cùng dùng CLI |
| `tests/` | Kiểm thử chia theo module |
| `docs/` | Hướng dẫn theo tác vụ |

Dữ liệu giữ nguyên: `Invoice High Accuracy v2/` (nguồn), `data_split/` (nguồn
đã chia), `data/` (ảnh), `groundtruth/` (nhãn), `outputs/` (prediction/báo cáo).
`gemini/outputs/` và `judge/outputs/` nếu tồn tại chỉ là cache cũ, không chứa
code ứng dụng. Không đưa dữ liệu khách hàng, `.env` hoặc `.env.labeling` lên Git.

## Tài liệu

- [Kiến trúc và vị trí file](docs/architecture.md)
- [Chạy Ollama local](docs/local.md)
- [GPU local, SSH, Kaggle và Colab](docs/gpu.md)
- [PostgreSQL](docs/database.md)
- [Gán nhãn và benchmark valid](docs/benchmark.md)

## Kiểm thử

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -t .
```

Test model/DB dùng mô phỏng; test giao diện phục vụ trang HTTP local. Kiểm thử
dịch vụ thật cần Ollama, model và PostgreSQL đang hoạt động. Model nhỏ trên CPU
dành cho thử nghiệm; độ chính xác cần đo bằng tập valid của dự án.
