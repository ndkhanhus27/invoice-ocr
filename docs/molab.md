# Molab — Gemma 4 31B bản gốc BF16

Chạy `notebooks/molab_gemma4.ipynb`. Notebook ngắn, lấy
`scripts/molab_benchmark.py` từ GitHub. Phiên bản này dùng **Transformers +
PyTorch**, model chính thức **google/gemma-4-31B-it**, dtype **BF16**.
Không cài/chạy Ollama, không tải GGUF và không lượng tử hóa.

## Chạy từng bước

1. Push notebook, script, backend và requirements mới lên GitHub.
2. Import lại file `.ipynb` mới trên Molab. Bản cũ có tên model Ollama sẽ bị từ chối.
3. Chọn Linux GPU, Python >= 3.12. Chạy cell cấu hình, giữ valid/limit=3 để thử.
4. Chạy cell lấy script từ GitHub rồi cell cài thư viện. Requirements lấy cùng commit.
5. Chạy cell benchmark, nhập HF token qua ô ẩn hoặc đặt secret HF_TOKEN.
6. Chạy cell kết quả và tải ZIP. Đổi limit=0 khi muốn chạy hết phần reviewed.

Repo code private: GITHUB_PRIVATE=True, nhập GitHub token khi được hỏi hoặc đặt
secret GITHUB_TOKEN. Không dán token vào source, không gửi token trong chat.

## Kiểm tra phần cứng trước khi tải weights

32 GiB hiển thị trên Molab là RAM hệ thống; VRAM được đo từ PyTorch CUDA.
RTX PRO 6000 của Molab được công bố có 96 GB VRAM. Script kiểm tra cuda:0 và yêu cầu
BF16 cùng tối thiểu 80 GiB VRAM còn trống. Đây là ngưỡng sơ bộ cho cấu hình mặc định,
không bảo đảm mọi hóa đơn/context đều vừa. Nếu chỉ có GPU 32 GiB VRAM, script dừng
trước khi tải model. Không tự chuyển model hoặc offload lên CPU/disk.

Nguồn Google ước tính Gemma 4 31B BF16 cần khoảng 69.9 GB cho weights/loading;
context và phần chạy có thể cần thêm. HF weights gốc có thể cần hơn 60 GB disk;
script dùng metadata repo để kiểm tra dung lượng trước khi tải. Cache Ollama cũ
vẫn được giữ nguyên, không tự xóa; dung lượng đó có thể làm thiếu chỗ tải bản mới.

Nguồn tham khảo:
- https://gemma.ai/models/gemma-4/#gemma-4-inference-memory-requirements
- https://huggingface.co/google/gemma-4-31B-it
- https://docs.marimo.io/guides/molab/#compute

## Backend và cấu hình

requirements/molab.txt dành riêng runtime Molab: Transformers 5.24.0, Accelerate,
HF Hub, Safetensors và Pillow. Dùng CUDA torch/torchvision sẵn có của Molab;
kiểm tra trước khi cài, không tự thay CUDA wheel. Không cài requirements/gpu.txt
cũ vào runtime này: file đó phục vụ nhánh local checkpoint 4-bit.

Model được tải từ repo chính thức ở commit đã resolve; các shard safetensors
được đọc lần lượt bằng low_cpu_mem_usage=True, nạp một lần trên cuda:0 trong
tiến trình worker Python. Kernel notebook không chứa model. Không dùng auto
device map để tránh vô tình offload weights gốc sang RAM 32 GiB.

Mặc định batch 1, max_input_tokens=8192, max_tokens=4096, max_edge=1600,
vision_soft_tokens=1120 cho đọc chữ nhỏ. Thinking tắt, do_sample=False để dễ
so sánh các lần chạy. Nếu input vượt giới hạn, hóa đơn báo lỗi thay vì bỏ trang.
Nhiều trang/đầu ra dài vẫn có thể hết VRAM; tăng tham số phải kiểm tra bộ nhớ.
Giữ nguyên prompt, schema, pipeline và scorer. Backend local bnb4 vẫn có sẵn;
BF16 là lựa chọn rõ ràng qua --weights-format bf16, không thay ngầm local config.

## Dữ liệu, log và kết quả

Tải full dataset HF private vào cache. Chỉ chấm reviewed của split đã chọn.
Ưu tiên groundtruth/reviewed/<split>.jsonl; nếu thiếu file export, chỉ lấy dòng
review_status=reviewed không có editor_draft từ annotations. Không dùng nhãn draft.
Ảnh snapshot HF được nối bằng hardlink/copy để vượt lỗi symlink blobs, vẫn kiểm
tra đầy đủ hash và thứ tự ảnh. Nếu ảnh thực sự thiếu, báo tên file cần upload.

Code/dataset/model đều được ghim commit SHA trong mỗi run. Predictions được ghi
ngay từng hóa đơn bằng cơ chế atomic của pipeline. Log stdout/stderr hiện trực
tiếp, có heartbeat khi worker im lặng. Tiến trình bị lỗi/kill vẫn giữ file đã ghi;
các prediction lỗi/thiếu vẫn tính khi scorer chạy được.

Kết quả ở invoice-results/<run-id>/results/ ngay trong working directory:
predictions, groundtruth đã chọn, metrics.csv, errors.csv, run_manifest.json và log.
Run manifest được cập nhật theo từng dòng tiến độ. ZIP được tạo khi kết thúc
bình thường hoặc gặp exception xử lý được. Kernel/container bị kill cứng có thể
không tạo ZIP; khi file còn trên disk có thể tải cả thư mục qua file browser.

**Đây vẫn là file local trong runtime, không phải lưu trữ bền vững.** Molab thay
container có thể làm mất file. Tải kết quả về hoặc tự lưu vào remote storage để
giữ qua lần reset. Notebook không tự upload dữ liệu/kết quả ra dịch vụ khác.
Không tuyên bố đã xác định nguyên nhân lần mất output trước nếu chưa có OOM log.

Các notebook/file marimo .py cũ chưa xóa, không dùng trong luồng BF16 này.
