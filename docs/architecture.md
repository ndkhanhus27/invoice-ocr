# Kiến trúc và trách nhiệm

Giao diện và CLI gọi cùng một pipeline. Backend chỉ xử lý giao tiếp model;
pipeline chịu trách nhiệm toàn hóa đơn, hash ảnh, kiểm tra JSON và lưu prediction.

```text
app/ui.py → app/service.py → extraction/pipeline.py
                                 ├── backends/ollama.py
                                 └── backends/transformers.py
                                 ↓
                         contracts/invoice.py
                                 ↓
                       storage/files.py (JSON)
app/service.py → storage/postgres.py → PostgreSQL
```

```text
app/
  ui.py                          Gradio và hiển thị kết quả
  service.py                     Kiểm tra upload, gọi pipeline, lưu DB
src/invoice_ocr/
  cli.py, __main__.py             CLI thống nhất
  config.py                      .env và tham số dòng lệnh
  contracts/invoice.py           Schema JSON, kiểu dữ liệu, Decimal
  extraction/
    pipeline.py                  Nhận đủ ảnh, metadata, validate, prediction
    backends/ollama.py            HTTP đến Ollama; CPU/auto
    backends/transformers.py      CUDA setup/download và model adapter
  storage/
    files.py                     JSON chính xác, ảnh, hash, atomic write
    postgres.py                  Transaction và mapping JSON sang bảng
  evaluation/
    runner.py                    Đọc GT/prediction và xuất báo cáo
    scoring.py                   Exact match, ghép dòng, CER
  labeling/
    prepare.py                   Render dữ liệu thành ảnh, chuẩn bị nhãn
    review.py                    Server review HTML và cập nhật JSONL
    export.py                    Xuất reviewed/ZIP portable
    gemini.py                    Model A tạo nhãn từ PDF
    verify.py                    Model B đọc PDF độc lập rồi kiểm tra
    comparison.py                So sánh nhãn, trạng thái verification
scripts/
  start_local.ps1                Khởi động Windows
  setup_local.py                Cấu hình/dịch vụ và mở app
  setup_gpu.py                  Shortcut GPU setup
  split_dataset.ps1             Chia dữ liệu có nhóm hash để tránh leakage
database/schema.sql             5 bảng PostgreSQL và index
deploy/compose.yaml             PostgreSQL local
configs/*.env.example           Các profile mẫu
requirements/*.txt             Thư viện theo tác vụ
notebooks/kaggle.ipynb           Transformers trên Kaggle
notebooks/colab.ipynb            Transformers trên Colab
```

`__init__.py` đánh dấu các Python package. `pyproject.toml` khai báo package ở
`src/`, package `app` và entrypoint `invoice-ocr`; cài bằng `pip install -e .`
trong checkout. Các asset như prompt/SQL vẫn nằm trong checkout này.

## Cấu hình

App/extraction/DB dùng `.env`. Labeling dùng `.env.labeling`. Các file trong
`configs/` chỉ là mẫu; không chứa khóa thật. Với app, thứ tự ưu tiên là CLI →
biến môi trường → file env → mặc định. `.venv` chỉ chứa thư viện Python.

CPU và GPU qua Ollama dùng cùng code; đổi `INVOICE_OLLAMA_DEVICE=cpu` thành
`auto`. Đổi cả runtime sang Transformers bằng `INVOICE_BACKEND=transformers`
và cài `requirements/gpu.txt`/CUDA trên máy GPU.

## Dữ liệu sau refactor

Không đổi đường dẫn ảnh, PDF, JSONL và prediction có sẵn. Không sửa nội dung
nhãn vì di chuyển code. Cache gán nhãn cũ trong `gemini/outputs/` và
`judge/outputs/` tiếp tục được dùng nếu tồn tại để tránh gọi lại API. Checkout
mới dùng `outputs/labeling/generate/` và `outputs/labeling/verify/`.

Các file source cũ trong `scripts/`, `gemini/`, `judge/` đã chuyển vào package;
không giữ wrapper cho đường dẫn cũ. Notebook/ZIP cũ phải xuất lại. Nhánh CLI
`normalize` và module chuyển số dạng chuỗi đã bỏ; invoice-v3 trả số đúng kiểu
ngay từ extraction. Output cũ trên disk không bị xóa. DB vẫn giữ khả năng nhận
report cũ đã normalized để tương thích dữ liệu trước đây.
