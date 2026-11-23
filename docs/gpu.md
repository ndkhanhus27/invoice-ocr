# GPU, SSH và notebook

## Vẫn dùng Ollama khi có GPU

Trong `.env`, đổi:

```dotenv
INVOICE_OLLAMA_DEVICE=auto
```

Khởi động lại app. Cần GPU/driver được Ollama hỗ trợ. Dùng `ollama ps` trong lúc
model chạy để kiểm tra phân bổ CPU/GPU thực tế. Không cần sửa `.venv` để bật GPU
của Ollama. Model khác chọn bằng INVOICE_OLLAMA_MODEL rồi tải model đó.

## Ollama trên máy GPU của An

Máy An chạy Ollama và có model. Máy bạn chạy UI và PostgreSQL. Tạo tunnel trên
máy bạn (thay username/hostname/key bằng thông tin thật):

```powershell
ssh -i "$env:USERPROFILE\.ssh\invoice_gpu" -N -L 11435:127.0.0.1:11434 khanh@GPU_HOST
```

Giữ terminal tunnel, chỉnh `.env` trên máy bạn:

```dotenv
INVOICE_BACKEND=ollama
INVOICE_OLLAMA_URL=http://127.0.0.1:11435
INVOICE_OLLAMA_DEVICE=auto
```

Chạy app local như thường. Nếu dùng `-PullModel`, model được tải ở server phía
bên kia tunnel. Không cần mở PostgreSQL ra mạng hay thay code giao diện.

## Backend Transformers

Trên máy GPU NVIDIA, cài editable package rồi chuẩn bị CUDA:

```text
python -m pip install -e .
python scripts/setup_gpu.py
python -m invoice_ocr gpu doctor
python -m invoice_ocr gpu download
```

Đặt cấu hình theo `configs/transformers.env.example` trong `.env`, hoặc chỉ
định profile bằng `--env-file`. Đừng ghi đè credentials nếu `.env` đã có sẵn.
Backend này dùng checkpoint BitsAndBytes 4-bit; model mặc định nằm trong config.
Setup kiểm tra GPU trước khi tải/cài CUDA, sau đó cài `requirements/gpu.txt`.

```dotenv
INVOICE_BACKEND=transformers
INVOICE_MODEL_DIR=models/YOUR_BNB4_CHECKPOINT
INVOICE_DEVICE=cuda:0
INVOICE_OFFLINE=true
```

```text
python -m invoice_ocr extract --backend transformers --input groundtruth/reviewed/valid.jsonl --output outputs/gpu-valid-v1/raw
python -m invoice_ocr ui
```

Ollama và Transformers dùng chung schema, pipeline và DB. Không trộn output
nhiều lần chạy vào cùng thư mục benchmark. Thiếu VRAM không được tự bỏ ảnh;
giảm MAX_EDGE rồi đo lại chất lượng, hoặc dùng GPU có đủ bộ nhớ.

## Kaggle và Colab

Ở local xuất ZIP mới sau refactor:

```text
python -m invoice_ocr label export --split valid --zip
```

- `notebooks/kaggle.ipynb`: upload ZIP thành Dataset private, Add Input, bật GPU/Internet.
- `notebooks/colab.ipynb`: upload ZIP vào MyDrive/invoice-ocr, chọn GPU runtime.

Notebook cài `pyproject.toml` bằng editable install, rồi gọi CLI gpu/extract/
evaluate. ZIP chỉ kèm code, cấu hình mẫu và ảnh/nhãn đã reviewed của split đó;
không kèm `.env`, `.env.labeling`, checkpoint hay nhãn chưa duyệt. Xuất lại ZIP
khi đổi code/prompt/nhãn. ZIP và notebook trước refactor không dùng entrypoint mới.
