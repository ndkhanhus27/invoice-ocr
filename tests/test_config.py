import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
from invoice_ocr.config import ROOT
from invoice_ocr import config
from invoice_ocr.extraction.backends import transformers as gpu


class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "tmp").mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / "tmp")
        self.addCleanup(self.temp.cleanup)
        self.env = Path(self.temp.name) / "run.env"
        self.env.write_text('INVOICE_MODEL_ID=org/other-bnb-4bit\nINVOICE_MAX_EDGE=1200\n'
                            'INVOICE_OFFLINE=true\nINVOICE_INPUT_DIR="data with spaces/valid"\n', encoding="utf-8")
        self.clean_env = patch.dict(os.environ, {}, clear=True)
        self.clean_env.start()
        self.addCleanup(self.clean_env.stop)

    def test_file_configuration_and_derived_model_path(self):
        args = config.parse_args(["--env-file", str(self.env), "extract"])
        self.assertEqual(args.max_edge, 1200)
        self.assertEqual(args.input, ROOT / "data with spaces/valid")
        self.assertEqual(args.model, str(ROOT / "models/other-bnb-4bit"))
        self.assertTrue(args.offline)
        self.assertIsNone(args.limit)

    def test_cli_overrides_process_overrides_file(self):
        with patch.dict(os.environ, {"INVOICE_MAX_EDGE": "1300"}):
            args = config.parse_args(["extract", "--env-file", str(self.env)])
            self.assertEqual(args.max_edge, 1300)
            args = config.parse_args(["extract", "--env-file", str(self.env), "--max-edge", "1400", "--no-offline"])
            self.assertEqual(args.max_edge, 1400)
            self.assertFalse(args.offline)

    def test_download_uses_configured_model_id(self):
        args = config.parse_args(["gpu", "download", "--env-file", str(self.env)])
        fake_download = unittest.mock.Mock(return_value="local-model-folder")
        with patch.dict("sys.modules", {"huggingface_hub": types.SimpleNamespace(snapshot_download=fake_download)}):
            gpu.download(args)
        self.assertEqual(fake_download.call_args.kwargs["repo_id"], "org/other-bnb-4bit")
        self.assertEqual(fake_download.call_args.kwargs["local_dir"], str(ROOT / "models/other-bnb-4bit"))

    def test_bf16_and_image_budget_are_explicit(self):
        args = config.parse_args(["extract", "--env-file", str(self.env), "--weights-format", "bf16",
                                  "--max-input-tokens", "8192", "--vision-soft-tokens", "1120"])
        self.assertEqual(args.weights_format, "bf16")
        self.assertEqual(args.max_input_tokens, 8192)
        self.assertEqual(args.vision_soft_tokens, 1120)

    def test_explicit_missing_file_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Config file does not exist"):
            config.parse_args(["plan", "--env-file", str(self.env.parent / "missing.env")])

    def test_malformed_line_is_rejected(self):
        self.env.write_text("not a config assignment\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Invalid config line"):
            config.read_env(self.env)
