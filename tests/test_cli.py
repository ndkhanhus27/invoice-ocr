"""CLI routing does not depend on CUDA, a model server or a database."""
from types import SimpleNamespace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from invoice_ocr.cli import main
from invoice_ocr.config import parse_args
from invoice_ocr.storage.files import write_json, write_jsonl, images_digest
from tests.contracts.test_invoice import invoice


class CLITests(unittest.TestCase):
    def test_ui_accepts_env_before_command(self):
        runner = Mock(return_value=None)
        with patch('invoice_ocr.cli.importlib.import_module', return_value=SimpleNamespace(main=runner)):
            self.assertEqual(main(['--env-file', 'custom.env', 'ui', '--port', '7861']), 0)
        runner.assert_called_once_with(['--port', '7861', '--env-file', 'custom.env'])

    def test_label_generate_and_verify_receive_shared_config(self):
        for action in ('generate', 'verify'):
            runner = Mock(return_value=0)
            with patch('invoice_ocr.cli.importlib.import_module', return_value=SimpleNamespace(main=runner)):
                self.assertEqual(main(['label', action, '--split', 'valid', '--env-file', 'labels.env']), 0)
            runner.assert_called_once_with(['--split', 'valid', '--env-file', 'labels.env'])

    def test_db_and_gpu_have_explicit_actions(self):
        self.assertEqual(parse_args(['db', 'init']).db_action, 'init')
        self.assertEqual(parse_args(['gpu', 'doctor']).gpu_action, 'doctor')

    def test_removed_normalize_command_is_rejected(self):
        with patch('sys.stderr'), self.assertRaises(SystemExit) as error:
            parse_args(['normalize'])
        self.assertEqual(error.exception.code, 2)

    def test_evaluate_runs_through_cli_and_writes_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = [{'file': 'page_001.jpg', 'sha256': 'a'*64}]
            write_jsonl(root/'labels.jsonl', [{'document_id': 'doc', 'images': images, 'annotation': invoice()}])
            write_json(root/'raw/doc.json', {'label_format': 'invoice-v3', 'document_id': 'doc',
                'status': 'ok', 'images': images, 'images_sha256': images_digest(images), 'extraction': invoice()})
            self.assertEqual(main(['evaluate', '--groundtruth', str(root/'labels.jsonl'),
                                   '--predictions', str(root/'raw'), '--output', str(root/'report')]), 0)
            self.assertTrue((root/'report/metrics.csv').is_file())
            self.assertTrue((root/'report/errors.csv').is_file())
