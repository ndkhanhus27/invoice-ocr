"""Optional UI HTTP smoke test when requirements/local.txt is installed."""
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(importlib.util.find_spec('gradio'), 'Install requirements/local.txt for UI checks')
class UploadUITests(unittest.TestCase):
    def test_upload_page_and_config_are_served_without_model_or_torch(self):
        import gradio as gr
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from invoice_ocr.config import parse_args
        from app.service import UploadService
        from app.ui import build_ui
        args = parse_args(['--env-file', str(ROOT/'configs/ollama.env.example'), 'extract'])
        ui = build_ui(UploadService(args))
        app = gr.mount_gradio_app(FastAPI(), ui, path='/')
        with TestClient(app) as client:
            response = client.get('/')
            self.assertEqual(response.status_code, 200)
            self.assertIn('Invoice OCR', response.text)
            config = client.get('/config').json()
            self.assertTrue(any(c['type'] == 'file' and c['props'].get('file_count') == 'multiple'
                                for c in config['components']))
            self.assertTrue(any(c['type'] == 'dataframe' for c in config['components']))
            self.assertTrue(any(c['type'] == 'button' and c['props'].get('value') == 'Trích xuất'
                                for c in config['components']))
        ui.close()


if __name__ == '__main__':
    unittest.main()
