"""Transaction, idempotency and exact-value checks using a recording connection."""
from contextlib import contextmanager
from decimal import Decimal
import types
import unittest
from unittest.mock import patch

from invoice_ocr.config import ROOT
from invoice_ocr.storage import postgres
from invoice_ocr.storage.files import images_digest
from tests.contracts.test_invoice import invoice


def report():
    images = [{'file': 'page_001.png', 'sha256': 'a'*64}, {'file': 'page_002.png', 'sha256': 'b'*64}]
    return {'label_format': 'invoice-v3', 'document_id': 'doc', 'source': 'upload/doc',
            'images': images, 'images_sha256': images_digest(images), 'model': 'test', 'status': 'ok',
            'extraction': invoice()}


class Connection:
    def __init__(self, results=(), fail_items=False):
        self.results = iter(results)
        self.calls = []
        self.fail_items = fail_items
        self.rolled_back = False
    @contextmanager
    def transaction(self):
        try:
            yield
        except Exception:
            self.rolled_back = True
            raise
    def cursor(self): return self
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def execute(self, sql, params=None):
        self.calls.append((sql, params))
        if self.fail_items and 'INSERT INTO invoice_ocr.invoice_items' in sql:
            raise ValueError('Simulated DB failure')
    def fetchone(self): return next(self.results)


class DatabaseTests(unittest.TestCase):
    def setUp(self):
        module = types.ModuleType('psycopg.types.json')
        module.Jsonb = lambda value, **kwargs: value
        self.adapter = patch.dict('sys.modules', {'psycopg.types.json': module})
        self.adapter.start()
        self.addCleanup(self.adapter.stop)

    def test_multi_image_document_has_one_invoice_and_exact_items(self):
        connection = Connection([(1,), (2,), (3,)])
        raw = report()
        self.assertEqual(postgres.store_report(connection, raw), 'loaded')
        invoices = [p for sql, p in connection.calls if 'INSERT INTO invoice_ocr.invoices ' in sql]
        self.assertEqual(len(invoices), 1)
        self.assertEqual(invoices[0][3:5], [1, 2])
        item = next(p for sql, p in connection.calls if 'INSERT INTO invoice_ocr.invoice_items' in sql)
        self.assertEqual(item[-1], Decimal('12.3456'))
        run = next(p for sql, p in connection.calls if 'INSERT INTO invoice_ocr.extraction_runs' in sql)
        self.assertEqual(run[6], raw)

    def test_repeat_save_does_not_create_an_invoice(self):
        connection = Connection([(1,), None])
        self.assertEqual(postgres.store_report(connection, report()), 'already_loaded')
        self.assertFalse(any('INSERT INTO invoice_ocr.invoices ' in sql for sql, _ in connection.calls))

    def test_failed_item_rolls_back_transaction(self):
        connection = Connection([(1,), (2,), (3,)], fail_items=True)
        with self.assertRaises(ValueError): postgres.store_report(connection, report())
        self.assertTrue(connection.rolled_back)

    def test_failed_extraction_and_changed_image_hash_are_rejected(self):
        for field, value in [('status', 'error'), ('images_sha256', '0'*64)]:
            raw = report()
            raw[field] = value
            with self.assertRaises(ValueError): postgres.store_report(Connection(), raw)

    def test_schema_is_loaded_from_database_directory(self):
        connection = Connection()
        with patch.object(postgres, 'postgres_connection', return_value=connection):
            self.assertEqual(postgres.initialize_database(types.SimpleNamespace()), 0)
        self.assertTrue(any('CREATE TABLE IF NOT EXISTS invoice_ocr.invoices' in sql for sql, _ in connection.calls))
