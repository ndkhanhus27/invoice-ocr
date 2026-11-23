CREATE SCHEMA IF NOT EXISTS invoice_ocr;
CREATE TABLE IF NOT EXISTS invoice_ocr.documents (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    sha256 CHAR(64) NOT NULL UNIQUE,
    source_uri TEXT NOT NULL,
    page_count INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS invoice_ocr.extraction_runs (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    document_id BIGINT NOT NULL REFERENCES invoice_ocr.documents(id),
    run_sha256 CHAR(64) NOT NULL UNIQUE,
    model_name TEXT,
    model_revision TEXT,
    prompt_sha256 TEXT,
    normalization_version TEXT NOT NULL,
    raw_result JSONB NOT NULL,
    normalized_result JSONB NOT NULL,
    review_status TEXT NOT NULL CHECK (review_status IN ('pending','needs_review')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS invoice_ocr.invoices (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    document_id BIGINT NOT NULL REFERENCES invoice_ocr.documents(id),
    extraction_run_id BIGINT NOT NULL REFERENCES invoice_ocr.extraction_runs(id),
    invoice_index INTEGER NOT NULL DEFAULT 1,
    page_start INTEGER,
    page_end INTEGER,
    company_name TEXT,
    bill_to TEXT,
    ship_to TEXT,
    invoice_number TEXT,
    invoice_date DATE,
    purchase_order TEXT,
    currency TEXT CHECK (currency ~ '^[A-Z]{3}$'),
    vin TEXT,
    licence_plate TEXT,
    vehicle_year SMALLINT,
    vehicle_make TEXT,
    vehicle_model TEXT,
    vehicle_unit TEXT,
    odometer NUMERIC(18,3),
    odometer_unit TEXT CHECK (odometer_unit IN ('km', 'mi')),
    driver_licence TEXT,
    subtotal NUMERIC(18,4),
    total_payment NUMERIC(18,4),
    review_status TEXT NOT NULL CHECK (review_status IN ('pending','needs_review','approved')),
    review_details JSONB NOT NULL,
    UNIQUE (extraction_run_id, invoice_index)
);
CREATE UNIQUE INDEX IF NOT EXISTS one_approved_invoice_per_document
    ON invoice_ocr.invoices (document_id, invoice_index) WHERE review_status = 'approved';
CREATE TABLE IF NOT EXISTS invoice_ocr.invoice_items (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    invoice_id BIGINT NOT NULL REFERENCES invoice_ocr.invoices(id),
    line_number INTEGER NOT NULL,
    source_page INTEGER,
    item_code TEXT,
    description TEXT,
    quantity NUMERIC(18,6),
    unit_price NUMERIC(18,6),
    line_total NUMERIC(18,4),
    UNIQUE (invoice_id, line_number)
);
CREATE TABLE IF NOT EXISTS invoice_ocr.invoice_taxes (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    invoice_id BIGINT NOT NULL REFERENCES invoice_ocr.invoices(id),
    tax_number INTEGER NOT NULL,
    label TEXT,
    rate NUMERIC(12,8),
    amount NUMERIC(18,4),
    UNIQUE (invoice_id, tax_number)
);

ALTER TABLE invoice_ocr.invoices ADD COLUMN IF NOT EXISTS odometer_unit TEXT CHECK (odometer_unit IN ('km', 'mi'));

ALTER TABLE invoice_ocr.invoices ADD COLUMN IF NOT EXISTS currency TEXT CHECK (currency ~ '^[A-Z]{3}$');

CREATE INDEX IF NOT EXISTS extraction_runs_document_idx ON invoice_ocr.extraction_runs(document_id);
CREATE INDEX IF NOT EXISTS invoices_document_idx ON invoice_ocr.invoices(document_id);
CREATE INDEX IF NOT EXISTS invoices_purchase_order_idx ON invoice_ocr.invoices(purchase_order);
CREATE INDEX IF NOT EXISTS invoices_invoice_number_idx ON invoice_ocr.invoices(invoice_number);
