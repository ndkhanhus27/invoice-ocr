"""PostgreSQL adapter. Transaction boundaries and SQL stay in this module."""
from __future__ import annotations
from ..config import ROOT

from datetime import date
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re

from ..contracts.invoice import NORMALIZATION_VERSION, VEHICLE_KEYS, NUMERIC_ITEM_KEYS, validate_invoice
from .files import dumps_exact, images_digest

def postgres_connection(args):
    if not args.database_url:
        raise ValueError("Set INVOICE_DATABASE_URL in your local .env or process environment; credentials are not bundled into ZIP")
    import psycopg
    try:
        return psycopg.connect(args.database_url, autocommit=True, connect_timeout=10)
    except psycopg.Error:
        raise RuntimeError("PostgreSQL connection failed. Check INVOICE_DATABASE_URL, network access and credentials.") from None

def initialize_database(args):
    with postgres_connection(args) as connection:
        with connection.transaction():
            for statement in (ROOT / 'database/schema.sql').read_text(encoding='utf-8').split(';'):
                if statement.strip():
                    connection.execute(statement)
    print("Initialized invoice_ocr schema (5 tables).")
    return 0

def database_decimal(value):
    if value is None:
        return None
    if isinstance(value, (int, Decimal)) and not isinstance(value, bool):
        number = Decimal(value)
        if not number.is_finite():
            raise ValueError("Non-finite database number")
        return number
    if not isinstance(value, str) or not re.fullmatch(r"-?\d+(?:\.\d+)?", value):
        raise ValueError("Database numbers must be normalized decimal strings or null")
    return Decimal(value)


def store_report(connection, report):
    from psycopg.types.json import Jsonb
    direct = report.get('label_format') == 'invoice-v3'
    if direct:
        raw = report
        if raw.get('status') != 'ok':
            raise ValueError('Cannot load a failed invoice extraction')
        validate_invoice(raw['extraction'])
        count = len(raw.get('images', []))
        if not count or raw.get('images_sha256') != images_digest(raw['images']):
            raise ValueError('Missing or inconsistent invoice image identity')
        report = {'raw_result': raw, 'normalization': {'version': NORMALIZATION_VERSION},
                  'review_status': 'pending', 'pages': [{'page': 1, 'source_page_count': count,
                  'normalized': raw['extraction'], 'issues': []}], 'issues': []}
    else:
        raw = report['raw_result']
    if report.get("normalization", {}).get("version") != NORMALIZATION_VERSION:
        raise ValueError("Unsupported normalization version")
    sha = (raw.get("images_sha256") if direct else raw.get("source_sha256", "")).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", sha):
        raise ValueError("Original source_sha256 is required for document identity")
    pages = report["pages"] if direct else raw.get("pages", [])
    page_counts = [page.get("source_page_count") for page in pages if isinstance(page.get("source_page_count"), int)]
    page_count = max(page_counts) if page_counts else None
    digest = hashlib.sha256(dumps_exact(report).encode()).hexdigest()
    review = "needs_review" if report.get("review_status") == "needs_review" else "pending"
    snapshot = {key: value for key, value in report.items() if key != "raw_result"}
    # All rows for this document/run commit together, including original JSON.
    with connection.transaction():
        with connection.cursor() as cursor:
            cursor.execute("INSERT INTO invoice_ocr.documents (sha256,source_uri,page_count) VALUES (%s,%s,%s) ON CONFLICT (sha256) DO NOTHING RETURNING id",
                           (sha, raw.get("source") or "unknown", page_count))
            row = cursor.fetchone()
            if row:
                document_id = row[0]
            else:
                cursor.execute("SELECT id FROM invoice_ocr.documents WHERE sha256=%s", (sha,))
                document_id = cursor.fetchone()[0]
            cursor.execute("""INSERT INTO invoice_ocr.extraction_runs
                (document_id,run_sha256,model_name,model_revision,prompt_sha256,normalization_version,raw_result,normalized_result,review_status)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (run_sha256) DO NOTHING RETURNING id""",
                (document_id, digest, raw.get("model"), raw.get("model_revision"), raw.get("prompt_sha256"),
                 NORMALIZATION_VERSION, Jsonb(raw, dumps=dumps_exact) if direct else Jsonb(raw), Jsonb(snapshot, dumps=dumps_exact) if direct else Jsonb(snapshot), review))
            row = cursor.fetchone()
            if row is None:
                return "already_loaded"
            run_id = row[0]
            # No page-to-invoice assumptions for unmerged multi-page or failed files.
            if (page_count != 1 and not direct) or raw.get("status") != "ok" or len(report["pages"]) != 1 or not report["pages"][0].get("normalized"):
                return "staged_for_review"
            page = report["pages"][0]
            data = page["normalized"]
            vehicle = data["vehicle_information"]
            columns = ["document_id", "extraction_run_id", "invoice_index", "page_start", "page_end",
                       "company_name", "bill_to", "ship_to", "invoice_number", "invoice_date", "purchase_order", "currency",
                       "vin", "licence_plate", "vehicle_year", "vehicle_make", "vehicle_model", "vehicle_unit",
                       "odometer", "odometer_unit", "driver_licence", "subtotal", "total_payment", "review_status", "review_details"]
            values = [document_id, run_id, 1, 1 if direct else page["page"], page_count if direct else page["page"], data["company_name"], data["bill_to"], data["ship_to"],
                      data["invoice_id"], date.fromisoformat(data["date"]) if data["date"] else None, data["purchase_order"], data["currency"],
                      *(vehicle[key] for key in VEHICLE_KEYS), database_decimal(data["totals"]["subtotal"]),
                      database_decimal(data["totals"]["total_payment"]), review,
                      Jsonb(report.get("issues", []) + page.get("issues", []))]
            cursor.execute("INSERT INTO invoice_ocr.invoices (" + ",".join(columns) + ") VALUES (" + ",".join(["%s"] * len(columns)) + ") RETURNING id", values)
            invoice_id = cursor.fetchone()[0]
            for index, item in enumerate(data["items"], 1):
                cursor.execute("""INSERT INTO invoice_ocr.invoice_items
                    (invoice_id,line_number,source_page,item_code,description,quantity,unit_price,line_total)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
                    (invoice_id, index, None if direct else page["page"], item["item_code"], item["description"],
                     *(database_decimal(item[key]) for key in NUMERIC_ITEM_KEYS)))
            for index, tax in enumerate(data["totals"]["taxes"], 1):
                cursor.execute("INSERT INTO invoice_ocr.invoice_taxes (invoice_id,tax_number,label,rate,amount) VALUES (%s,%s,%s,%s,%s)",
                               (invoice_id, index, tax["label"], database_decimal(tax["rate"]), database_decimal(tax["amount"])))
    return "loaded"
