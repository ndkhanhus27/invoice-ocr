"""Portable ZIPs contain the new source layout and no private configuration."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from invoice_ocr.labeling import export as exporter
from invoice_ocr.storage import files
from tests.contracts.test_invoice import invoice


class ExportTests(unittest.TestCase):
    def test_zip_contains_installable_code_reviewed_images_and_no_secrets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('src/invoice_ocr/cli.py', 'app/ui.py', 'scripts/setup_gpu.py', 'database/schema.sql',
                         'requirements/gpu.txt', 'configs/transformers.env.example', 'prompts/invoice.txt', 'pyproject.toml'):
                target = root/name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text('test')
            (root/'.env').write_text('PRIVATE_SECRET=do-not-export')
            (root/'.env.labeling').write_text('GEMINI_API_KEY=do-not-export')
            image = root/'data/valid/doc/page_001.jpg'
            image.parent.mkdir(parents=True)
            image.write_bytes(b'original-image')
            labels = root/'groundtruth'
            records = [{'document_id': 'doc', 'images': ['data/valid/doc/page_001.jpg'], 'annotation': invoice()}]
            files.write_jsonl(labels/'reviewed/valid.jsonl', records)
            with patch.object(exporter, 'ROOT', root), patch.object(files, 'ROOT', root):
                archive = exporter.package(labels, 'valid')
            with ZipFile(archive) as zipped:
                names = zipped.namelist()
                self.assertIn('src/invoice_ocr/cli.py', names)
                self.assertIn('requirements/gpu.txt', names)
                self.assertIn('pyproject.toml', names)
                self.assertIn('groundtruth/reviewed/valid.jsonl', names)
                self.assertNotIn('.env', names)
                self.assertNotIn('.env.labeling', names)
                self.assertEqual(zipped.read('data/valid/doc/page_001.jpg'), b'original-image')
