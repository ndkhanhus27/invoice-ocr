# PostgreSQL

DDL nằm tại `database/schema.sql`, code kết nối/ghi DB tại
`src/invoice_ocr/storage/postgres.py`. `db init` chạy lại được và không xóa dữ liệu.

## Chạy local

Cách tự động: startup với `-StartDatabase -InitializeDatabase`. Script sinh
password/URL vào `.env` nếu chưa có. Nếu chạy Docker thủ công, đặt
INVOICE_POSTGRES_PASSWORD trong `.env`, rồi:

Nếu máy có dịch vụ khác ở cổng 5432, đặt `INVOICE_POSTGRES_PORT=55432` và dùng
cổng 55432 trong `INVOICE_DATABASE_URL`. PostgreSQL trong container vẫn dùng 5432.

```powershell
docker compose -p invoice-ocr --env-file .env -f deploy/compose.yaml up -d --wait postgres
```

Trong `.env`:

```dotenv
INVOICE_DATABASE_URL=postgresql://invoice:URL_ENCODED_PASSWORD@127.0.0.1:5432/invoice_ocr
```

```text
python -m invoice_ocr db init
python -m invoice_ocr db load --input outputs/ollama-valid-v1/raw
```

Nếu có DB sẵn trên máy khác, chỉ đổi connection string; không cần Docker local.
Mật khẩu có ký tự đặc biệt phải URL-encode trong connection string.

Compose giữ project name `invoice-ocr` dù file chuyển vào deploy/, để tái sử
dụng volume của cấu trúc cũ. Dữ liệu Docker nằm trong named volume. Lệnh down
không kèm `-v` sẽ giữ volume. Thay password trong `.env` không tự đổi password
của PostgreSQL đã khởi tạo; dùng password cũ hoặc đổi trong DB trước.

## Quan hệ bảng

```mermaid
erDiagram
    documents ||--o{ extraction_runs : has
    documents ||--o{ invoices : has
    extraction_runs ||--o{ invoices : produces
    invoices ||--o{ invoice_items : contains
    invoices ||--o{ invoice_taxes : contains
```

| Bảng trong schema invoice_ocr | Nội dung |
| --- | --- |
| documents | Hash nội dung ảnh theo thứ tự, đường dẫn nguồn, số trang |
| extraction_runs | Model/revision, prompt hash, JSON gốc và snapshot đã kiểm tra |
| invoices | Company/bill/ship, số hóa đơn, ngày, PO, tiền tệ, xe, subtotal, tổng tiền |
| invoice_items | Mỗi dòng hàng: mã, mô tả, số lượng, đơn giá, thành tiền |
| invoice_taxes | Mỗi loại thuế: tên, thuế suất, số tiền |

JSON `invoice_id` ánh xạ sang `invoices.invoice_number`; `invoices.id` là khóa
nội bộ. PO/mã hóa đơn là TEXT để giữ số 0 đầu. Tiền NUMERIC(18,4), số lượng/
đơn giá NUMERIC(18,6), thuế suất NUMERIC(12,8). Thiếu dữ liệu lưu NULL.

Một transaction lưu toàn bộ document/run/invoice/items/taxes. Lỗi thì rollback;
cùng kết quả lưu lại không sinh bản ghi mới. Model chạy lại là run riêng nếu
metadata/kết quả khác. Kết quả mới là pending, không tự coi là ground truth.
Tên cột normalization_version/normalized_result giữ lại để tương thích DB cũ;
pipeline invoice-v3 hiện tại validate trực tiếp, không qua bước normalize riêng.

Không cấu hình DB vẫn trích xuất được. DB không kết nối hoặc chưa tạo schema
thì nút lưu không thành công, nhưng JSON trên disk vẫn còn để nạp lại sau.
