"""Real HTTP request/response tests against a local fake Ollama server, no GPU."""
from __future__ import annotations

import base64
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
from invoice_ocr.config import parse_args
from invoice_ocr.extraction.pipeline import extract as extract_ollama
from invoice_ocr.storage.files import dumps_exact, loads_exact
from invoice_ocr.extraction.backends.ollama import OllamaClient, OllamaError
from invoice_ocr.contracts.invoice import TEXT_KEYS, VEHICLE_KEYS
from scripts.setup_local import ensure_config, update_env
from app.service import UploadService


def answer():
    return {**dict.fromkeys(TEXT_KEYS), 'purchase_order': '000042', 'currency': 'USD',
            'vehicle_information': dict.fromkeys(VEHICLE_KEYS),
            'items': [{'item_code': 'A', 'description': 'Service', 'quantity': 1,
                       'unit_price': Decimal('12.3456'), 'line_total': Decimal('12.3456')}],
            'totals': {'subtotal': Decimal('12.3456'), 'taxes': [], 'total_payment': Decimal('12.3456')}}


class OllamaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.requests = []
        cls.response = {}
        cls.capabilities = ['completion', 'vision']
        cls.http_status = 200
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_GET(self):
                self.send({'models': [{'name': 'qwen3-vl:2b', 'digest': 'sha256:test'}]})
            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                cls.requests.append((self.path, payload))
                self.send({'capabilities': cls.capabilities} if self.path == '/api/show' else cls.response)
            def send(self, data):
                encoded = json.dumps(data).encode()
                self.send_response(cls.http_status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        from PIL import Image
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.document = self.root/'invoice01'
        self.document.mkdir()
        for i in range(2):
            Image.new('RGB', (16, 16), (i*100, 0, 0)).save(self.document/f'page_{i+1:03}.png')
        self.args = SimpleNamespace(backend='ollama', input=self.document, output=self.root/'raw',
            limit=None, overwrite=False, prompt=ROOT/'prompts/invoice.txt', max_edge=100, max_new_tokens=4096,
            ollama_url=f'http://127.0.0.1:{self.server.server_port}', ollama_model='qwen3-vl:2b',
            ollama_timeout=10, ollama_device='cpu', ollama_num_ctx=8192,
            ollama_num_threads=None, ollama_keep_alive='5m')
        type(self).requests.clear()
        type(self).http_status = 200
        type(self).capabilities = ['completion', 'vision']
        type(self).response = {'done': True, 'done_reason': 'stop', 'eval_count': 100,
                              'message': {'content': dumps_exact(answer())}}

    def read_result(self):
        return loads_exact((self.args.output/'invoice01.json').read_text(encoding='utf-8'))

    def test_http_pipeline_preserves_order_decimal_and_full_schema_without_torch(self):
        self.assertEqual(extract_ollama(self.args), 0)
        result = self.read_result()
        self.assertEqual(result['extraction']['totals']['total_payment'], Decimal('12.3456'))
        self.assertEqual(result['extraction']['purchase_order'], '000042')
        self.assertEqual(result['model_revision'], 'sha256:test')
        self.assertEqual(len(result['images']), 2)
        payload = next(payload for path, payload in self.requests if path == '/api/chat')
        self.assertFalse(payload['stream'])
        self.assertEqual(payload['options']['num_gpu'], 0)
        self.assertIn('purchase_order', payload['format']['required'])
        from PIL import Image
        images = payload['messages'][0]['images']
        for index, data in enumerate(images):
            with Image.open(BytesIO(base64.b64decode(data))) as image:
                self.assertEqual(image.getpixel((0, 0)), (index*100, 0, 0))

    def test_auto_omits_cpu_override(self):
        self.args.ollama_device = 'auto'
        from PIL import Image
        with Image.new('RGB', (1, 1)) as image:
            OllamaClient(self.args).generate([image], 'test')
        self.assertNotIn('num_gpu', self.requests[-1][1]['options'])

    def test_bad_json_keeps_raw_text_but_cannot_be_saved(self):
        type(self).response['message']['content'] = 'not json'
        self.assertEqual(extract_ollama(self.args), 1)
        result = self.read_result()
        self.assertEqual(result['raw_text'], 'not json')
        self.assertNotIn('extraction', result)
        with self.assertRaises(ValueError):
            UploadService(self.args).save(result)

    def test_truncated_valid_json_is_still_rejected(self):
        type(self).response['done_reason'] = 'length'
        self.assertEqual(extract_ollama(self.args), 1)
        self.assertIn('Incomplete', self.read_result()['error'])

    def test_text_only_model_is_rejected(self):
        type(self).capabilities = ['completion']
        self.assertEqual(extract_ollama(self.args), 1)
        self.assertIn('does not support images', self.read_result()['error'])
        self.assertFalse(any(path == '/api/chat' for path, _ in self.requests))

    def test_missing_model_is_actionable(self):
        self.args.ollama_model = 'missing:2b'
        with self.assertRaisesRegex(OllamaError, 'ollama pull missing:2b'):
            OllamaClient(self.args).check_model()

    def test_timeout_and_http_error_do_not_expose_server_body(self):
        client = OllamaClient(self.args)
        with patch.object(client.opener, 'open', side_effect=TimeoutError):
            with self.assertRaisesRegex(OllamaError, 'timed out'):
                client.generate([], '')
        type(self).http_status = 500
        type(self).response = {'error': 'sensitive server data'}
        with self.assertRaisesRegex(OllamaError, 'HTTP 500') as error:
            client.generate([], '')
        self.assertNotIn('sensitive', str(error.exception))

    def test_upload_uses_ollama_and_produces_db_ready_record(self):
        self.args.database_url = None
        status, text, download, result = UploadService(self.args).run(
            list(map(str, sorted(self.document.iterdir()))))
        self.assertIn('000042', status)
        self.assertEqual(result['backend'], 'ollama')
        self.assertTrue(Path(download).is_file())
        self.assertEqual(loads_exact(text), answer())
        self.assertEqual(result['status'], 'ok')

    def test_config_cli_overrides_env_and_existing_config_is_preserved(self):
        path = self.root/'run.env'
        path.write_text('INVOICE_DATABASE_URL=postgresql://user:secret@localhost/db\nINVOICE_MODEL_ID=existing\n')
        with patch.dict('os.environ', {}, clear=True):
            ensure_config(path)
            update_env(path, {'INVOICE_OLLAMA_MODEL': 'qwen3-vl:2b'})
            args = parse_args(['--env-file', str(path), 'extract', '--ollama-device', 'auto'])
        self.assertEqual(args.backend, 'ollama')
        self.assertEqual(args.ollama_device, 'auto')
        self.assertEqual(args.database_url, 'postgresql://user:secret@localhost/db')
        self.assertIn('INVOICE_MODEL_ID=existing', path.read_text())


if __name__ == '__main__':
    unittest.main()
