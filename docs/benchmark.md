# Ground truth và benchmark valid

Một thư mục ảnh là một hóa đơn, một annotation chung, một prediction JSON.
Không chia các trang cùng hóa đơn sang nhiều split. document_id là mã nội bộ,
khác invoice_id in trên hóa đơn. Nhãn dùng các trường của contracts/invoice.py.

## Chuẩn bị và review

Nếu chỉ chạy valid đã có nhãn, có thể bỏ qua chuẩn bị và gán nhãn bằng API.
Khi cần render PDF/tạo dữ liệu:

```text
python -m pip install -r requirements/labeling.txt -e .
python -m invoice_ocr label prepare
python -m invoice_ocr label review
```

prepare đọc data_split/, tạo ảnh JPEG 200 DPI trong data/ và JSONL trong
groundtruth/annotations/. Giữ nhãn invoice-v3 có sẵn, báo lỗi nếu nguồn thay đổi.
Review server in địa chỉ local; kiểm tra đủ trang, sửa JSON rồi cập nhật. Nhãn
đã duyệt tự đồng bộ sang groundtruth/reviewed/. Không sửa trực tiếp bản trích
reviewed vì có thể tạo lại từ annotations.

## Gán nhãn bằng hai model (tùy chọn)

Copy `configs/labeling.env.example` thành `.env.labeling` nếu chưa có. Hai model
có key/model riêng trong cùng file. Các cấu hình API cũ đã được giữ lại trong
file này khi refactor. Có thể override bằng biến môi trường hoặc --env-file.

```text
python -m invoice_ocr label generate --split valid --limit 1 --only-unlabeled --apply
python -m invoice_ocr label verify --split valid --limit 1 --apply
```

Thêm `--plan` để liệt kê mà không gọi API. Hai model cùng đọc PDF gốc độc lập;
không gửi đáp án A cho B. Model A lưu draft; model B so kết quả, đồng ý hoàn toàn
và có nội dung mới được tự duyệt. Hai model vẫn có thể cùng sai; review thủ công
khi có khác biệt. Checker không ghi đè annotation A.

Cache cũ gemini/outputs và judge/outputs được tái sử dụng nếu thư mục tồn tại;
checkout mới lưu dưới outputs/labeling/generate và outputs/labeling/verify.
Không đổi model/prompt rồi trộn nhiều run để đo chung.

## Chạy model trên valid

```text
python -m invoice_ocr plan --input groundtruth/reviewed/valid.jsonl --output outputs/ollama-valid-v1/raw
python -m invoice_ocr extract --backend ollama --input groundtruth/reviewed/valid.jsonl --output outputs/ollama-valid-v1/raw --limit 1
```

Lệnh smoke test tạo prediction thật. Muốn chạy toàn bộ sau đó, chọn một thư mục
output mới và bỏ `--limit 1`. Không tự ghi đè kết quả. Dùng --backend transformers
khi chạy trên GPU với checkpoint/requirements đã chuẩn bị.

```text
python -m invoice_ocr evaluate --groundtruth groundtruth/reviewed/valid.jsonl --predictions outputs/ollama-valid-full/raw --output outputs/ollama-valid-full/evaluation
```

metrics.csv ghi điểm theo trường, items_rows, taxes_rows, invoice_all_fields,
schema_valid và CER. errors.csv ghi mã hóa đơn, trường, giá trị đúng/dự đoán và
lý do sai. Thiếu prediction, lỗi schema hoặc ảnh/hash sai vẫn tính là lỗi.

Số so bằng Decimal chính xác, không dung sai/làm tròn. Mã giữ số 0, chữ hoa/thường
và dấu câu, chỉ chuẩn hóa Unicode/khoảng trắng. Văn bản chuẩn hóa Unicode,
khoảng trắng, chữ hoa/thường và dấu nháy/gạch tương đương. null/null thống kê
riêng, không tăng điểm Exact Match. Exact Match = correct/(correct+wrong+missing+extra).
Dòng hàng ghép một-một bằng code/description, sau đó thứ tự dòng; không dùng
giá trị số để chọn cặp ghép. CER chỉ là chẩn đoán lỗi chữ, không công nhận giá
trị sai thành đúng. Dùng valid chỉnh prompt/tham số, giữ test cho lần đánh giá cuối.

## Xuất ZIP cho GPU

```text
python -m invoice_ocr label export --split valid --zip
```

Tạo invoice-reviewed-valid.zip gồm source mới, pyproject, requirements, prompt,
cấu hình mẫu và chỉ ảnh/nhãn reviewed của valid. Không kèm khóa API, .env,
checkpoint hay nhãn chưa duyệt. Chạy lại export sau mỗi lần đổi code/nhãn.
