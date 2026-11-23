# Benchmark Gemma 4 trên Molab

Notebook: `notebooks/molab_gemma4.py` (marimo Python, có dependency metadata).
Notebook không chạy khi chỉ mở file; mọi download/inference chỉ bắt đầu sau khi
submit form **Chạy benchmark**. Không cần PostgreSQL, Docker hoặc GPU trên máy cá nhân.

## 1. Đưa code lên GitHub bằng PowerShell

Tại thư mục dự án, kiểm tra nội dung chuẩn bị commit rồi push:

```powershell
git add -- .gitignore README.md pyproject.toml app configs database deploy docs notebooks prompts requirements scripts src tests
git diff --cached --stat
git diff --cached --name-only
git commit -m "Add invoice OCR pipeline and Molab benchmark notebook"
git push origin main
```

Không stage `.env`, token, data hoặc groundtruth. Dữ liệu đã upload riêng lên
Hugging Face dataset private `ndkhanhus27/invoice-ocr-data`.

## 2. Mở Molab

Chọn runtime Linux x86_64 có NVIDIA GPU (RTX PRO 6000). Import/upload file
`notebooks/molab_gemma4.py` từ máy hoặc từ GitHub nếu giao diện hỗ trợ import URL:

```text
https://raw.githubusercontent.com/ndkhanhus27/invoice-ocr/main/notebooks/molab_gemma4.py
```

Nếu repo private, tải file từ GitHub bằng tài khoản của bạn rồi upload file lên Molab.
Molab cần cài các dependency ghi ở đầu file. Nếu môi trường không tự đọc metadata,
cài trong terminal của chính runtime Molab trước khi chạy notebook:

```bash
python -m pip install 'marimo>=0.19,<1' 'huggingface-hub>=0.34,<2' 'Pillow>=10,<13' 'zstandard>=0.23,<1'
```

## 3. Cấu hình và chạy

- HF token: quyền đọc dataset private, nhập ô ẩn hoặc đặt secret `HF_TOKEN`.
- GitHub token: chỉ cần nếu repository code private, nhập ô ẩn hoặc secret `GITHUB_TOKEN`.
- Giữ split `valid` (dev đã review); không dùng annotations chưa review để chấm.
- Chạy thử 3 hóa đơn bằng `Số hóa đơn = 3`. Khi cần toàn bộ reviewed, đặt `0`.
- Model mặc định `gemma4:31b-it-q8_0` (Gemma 4 31B Q8, vision), không phải 40B.
- Bấm **Chạy benchmark**. Lần đầu tải model khoảng 34 GB, cùng toàn bộ dataset.
- Tải ZIP kết quả trước khi runtime bị thu hồi.

Notebook resolve GitHub/HF revision thành commit SHA trước khi tải; model digest
được ghi trong predictions và danh sách model của run manifest. Có thể nhập commit
SHA cho các lần chạy lặp lại. Ollama mặc định `latest` dùng lại bản đã tải trong
cache; version thực tế luôn được ghi. Để cố định binary, điền số release cụ thể.

Các tham số mặc định: context 16384, output 8192 tokens, ảnh tối đa 1600px, timeout
1800 giây/hóa đơn. Dùng cùng pipeline, prompt, JSON schema và scorer của code
GitHub đã chọn. Thinking mode theo mặc định model/Ollama; notebook không thay đổi
logic backend hoặc tự tinh chỉnh prompt dựa trên groundtruth.

## 4. Dữ liệu và kết quả

Dataset cần `data/`, `groundtruth/reviewed/<split>.jsonl`; `data_split/` và
`groundtruth/annotations/` được tải cùng nếu có. Dữ liệu giữ trong HF cache,
model trong `~/.cache/invoice-molab/models`; cache chỉ tồn tại khi runtime còn giữ disk.

Chỉ các hóa đơn trong reviewed được chọn. Nếu reviewed chưa phủ toàn bộ split,
bảng điểm chỉ đại diện phần reviewed đó. Limit lấy các dòng đầu theo thứ tự JSONL,
không chọn theo độ dễ. Groundtruth được giữ nguyên chuỗi JSON để không làm tròn
số; input cho extraction chỉ gồm ID và ảnh, không chứa đáp án.

ZIP bao gồm `run_manifest.json`, `groundtruth_selected.jsonl`, `input_images.jsonl`,
`predictions/*.json`, `evaluation/metrics.csv`, `evaluation/errors.csv` và log.
PO, `items_rows`, `totals.total_payment` được hiển thị riêng; bảng toàn bộ tiêu chí
cũng có sẵn. Lỗi/thiếu predictions vẫn nằm trong mẫu chấm. Thời gian trích xuất
được tách khỏi tổng thời gian tải/chuẩn bị. ZIP chứa nhãn và nội dung hóa đơn đã
trích xuất, nên lưu riêng cùng dữ liệu; notebook không tự upload kết quả.

Ollama chỉ lắng nghe localhost ở cổng riêng và được dừng sau mỗi lần chạy, kể cả
khi có lỗi. Không thao tác Ollama hay PostgreSQL trên máy Windows của bạn.
GPU preflight yêu cầu có VRAM sử dụng, không bảo đảm toàn bộ model nằm trên GPU;
kiểm tra `gpu_preflight` và server log nếu đo hiệu năng. Kiểm tra trống 44.000 MiB
cho model mặc định là ngưỡng sơ bộ, không phải cam kết đủ cho mọi hóa đơn/context.

Sau này cập nhật dataset trên HF và code trên GitHub rồi chạy lại notebook với
revision `main` để lấy phiên bản mới. Không cần sửa notebook nếu chỉ cập nhật nhãn.
