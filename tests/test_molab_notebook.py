"""Exercise notebook subprocess logging without marimo, network, data or model."""
import ast
import json
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from contextlib import nullcontext, redirect_stdout
from unittest.mock import patch


def load_stream_command():
    path = Path(__file__).resolve().parents[1] / "scripts/molab_benchmark.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef) and node.name == "stream_command")
    scope = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), scope)
    return scope["stream_command"]


class MolabLoggingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.stream = load_stream_command()

    def run_command(self, source, emit, **kwargs):
        return self.stream([sys.executable, "-u", "-c", source], self.directory / "command.log",
                           emit, cwd=self.directory, env=os.environ.copy(), **kwargs)

    def test_stdout_reaches_callback_before_child_finishes(self):
        # Child cannot finish until the parent's live callback acknowledges its first line.
        seen = []
        def emit(message):
            seen.append(message)
            if "ready" in message:
                (self.directory / "ack").touch()
        source = """import pathlib,time
print('ready', flush=True)
deadline=time.monotonic()+4
while not pathlib.Path('ack').exists():
    if time.monotonic()>deadline: raise RuntimeError('output was buffered')
    time.sleep(.01)
print('finished', flush=True)
"""
        self.assertEqual(self.run_command(source, emit, timeout=6), 0)
        self.assertTrue(any("finished" in message for message in seen))

    def test_stderr_and_nonzero_exit_are_visible_and_saved(self):
        seen = []
        code = self.run_command("import sys; print('model failed',file=sys.stderr); sys.exit(7)", seen.append)
        self.assertEqual(code, 7)
        self.assertTrue(any("model failed" in message for message in seen))
        self.assertIn("model failed", (self.directory / "command.log").read_text(encoding="utf-8"))
        self.assertIn("exit code = 7", seen[-1])

    def test_silent_child_has_heartbeat(self):
        seen = []
        self.assertEqual(self.run_command("import time; time.sleep(.3)", seen.append, heartbeat_seconds=.05), 0)
        self.assertTrue(any("vẫn đang chạy" in message for message in seen))

    def test_timeout_stops_child(self):
        with self.assertRaises(subprocess.TimeoutExpired):
            self.run_command("import time; time.sleep(30)", lambda message: None,
                             timeout=.1, heartbeat_seconds=.05)

    def test_start_message_is_visible_before_token_prompt(self):
        path = Path(__file__).resolve().parents[1] / "scripts/molab_benchmark.py"
        scope = {}
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), scope)
        scope["run_benchmark"] = lambda config, progress: {"status": "mocked"}
        captured = io.StringIO()
        def enter_token(prompt):
            self.assertIn("Đã nhận lệnh chạy benchmark", captured.getvalue())
            self.assertIn("Đang chờ nhập HF token", captured.getvalue())
            return "test-placeholder"
        with patch.dict(os.environ, {}, clear=True), patch("getpass.getpass", side_effect=enter_token), redirect_stdout(captured):
            self.assertEqual(scope["run_from_notebook"]({}), {"status": "mocked"})
        self.assertNotIn("test-placeholder", captured.getvalue())

    def test_notebook_only_loads_and_calls_github_script(self):
        path = Path(__file__).resolve().parents[1] / "notebooks/molab_gemma4.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        sources = []
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])
                source = "".join(cell["source"])
                ast.parse(source)
                sources.append(source)
        source = "\n".join(sources)
        self.assertLess(len(source.splitlines()), 60)
        self.assertIn("raw.githubusercontent.com", source)
        self.assertIn("/scripts/molab_benchmark.py", source)
        self.assertIn("benchmark.run_from_notebook(CONFIG", source)
        self.assertNotIn("def run_benchmark(", source)
        self.assertNotIn("import marimo", source)
        self.assertIn('"model": "google/gemma-4-31B-it"', source)
        self.assertIn('"weights_format": "bf16"', source)
        self.assertNotIn("ollama_version", source)

    def test_original_model_stops_on_32_gib_vram(self):
        path = Path(__file__).resolve().parents[1] / "scripts/molab_benchmark.py"
        scope = {}
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), scope)
        with self.assertRaisesRegex(RuntimeError, "80 GiB"):
            scope["require_bf16_hardware"]({"bf16": True, "vram_free_gib": 32})
        scope["require_bf16_hardware"]({"bf16": True, "vram_free_gib": 90})

    def test_old_ollama_config_is_not_silently_used(self):
        path = Path(__file__).resolve().parents[1] / "scripts/molab_benchmark.py"
        scope = {}
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), scope)
        with self.assertRaisesRegex(ValueError, "bản gốc"):
            scope["validate_bf16_config"]({"model": "gemma4:31b-it-q8_0"})


class MolabImageCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        path = Path(__file__).resolve().parents[1] / "scripts/molab_benchmark.py"
        scope = {}
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), scope)
        self.materialize = scope["materialize_image_tree"]
        self.verify = scope["verify_image_files"]

    def test_hf_blob_symlink_becomes_valid_invoice_manifest(self):
        from invoice_ocr.storage import files
        snapshot = self.root / "hub/snapshots/commit"
        blob = self.root / "hub/blobs/content"
        blob.parent.mkdir(parents=True)
        blob.write_bytes(b"image identity for plan")
        relative = "data/valid/invoice-1/page_001.jpg"
        cached_image = snapshot / relative
        cached_image.parent.mkdir(parents=True)
        resolution = nullcontext()
        try:
            cached_image.symlink_to(blob)
        except OSError:
            # Windows may forbid symlinks. Emulate only their resolve behavior;
            # destination hardlinks and the complete document planner remain real.
            cached_image.write_bytes(b"placeholder must not be copied")
            original_resolve = Path.resolve
            def resolve_cache_path(path, *args, **kwargs):
                return blob if path == cached_image else original_resolve(path, *args, **kwargs)
            resolution = patch.object(Path, "resolve", resolve_cache_path)
        checkout = self.root / "code"
        checkout.mkdir()
        with resolution:
            with patch.object(files, "ROOT", snapshot):
                with self.assertRaisesRegex(ValueError, "Invalid image path"):
                    files.image_entries([relative])
            self.materialize(snapshot / "data", checkout / "data", lambda message: None)
        self.assertFalse((checkout / relative).is_symlink())
        manifest = checkout / "selected.jsonl"
        manifest.write_text(json.dumps({"document_id": "invoice-1", "images": [relative]}) + "\n", encoding="utf-8")
        with patch.object(files, "ROOT", checkout):
            _, documents = files.documents(manifest)
            self.assertEqual(len(documents), 1)
            self.assertEqual(files.image_entries([relative])[0]["sha256"], files.file_sha256(blob))

    def test_cross_filesystem_uses_copy(self):
        source = self.root / "source"
        source.mkdir()
        (source / "page.jpg").write_bytes(b"original")
        destination = self.root / "destination"
        with patch("os.link", side_effect=OSError("different filesystem")):
            counts = self.materialize(source, destination, lambda message: None)
        self.assertEqual(counts, {"hardlinked": 0, "copied": 1})
        self.assertEqual((destination / "page.jpg").read_bytes(), b"original")
        self.assertFalse((destination / "page.jpg").is_symlink())

    def test_missing_upload_has_specific_diagnostic(self):
        records = [{"images": ["data/valid/missing/page_001.jpg"]}]
        with self.assertRaisesRegex(FileNotFoundError, "Upload các ảnh này"):
            self.verify(self.root, records)


class MolabGroundtruthTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.dataset = Path(self.temporary.name)
        path = Path(__file__).resolve().parents[1] / "scripts/molab_benchmark.py"
        scope = {}
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), scope)
        self.load = scope["load_reviewed_groundtruth"]

    def write(self, folder, lines):
        path = self.dataset / "groundtruth" / folder / "valid.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def test_missing_export_uses_only_explicit_reviewed_and_preserves_numbers(self):
        reviewed = '{"document_id":"yes","review_status":"reviewed","annotation":{"amount":123456789.123456789}}'
        self.write("annotations", [reviewed,
                   '{"document_id":"no","review_status":"needs_review"}',
                   '{"document_id":"unknown"}',
                   '{"document_id":"draft","review_status":"reviewed","editor_draft":{}}'])
        source, lines = self.load(self.dataset, "valid", lambda message: None)
        self.assertEqual(source.parent.name, "annotations")
        self.assertEqual(lines, [reviewed])

    def test_export_is_preferred(self):
        expected = '{"document_id":"exported","annotation":{}}'
        self.write("reviewed", [expected])
        self.write("annotations", ['{"review_status":"reviewed","document_id":"other"}'])
        source, lines = self.load(self.dataset, "valid", lambda message: None)
        self.assertEqual(source.parent.name, "reviewed")
        self.assertEqual(lines, [expected])

    def test_unreviewed_annotations_do_not_become_groundtruth(self):
        self.write("annotations", ['{"review_status":"needs_review","annotation":{}}'])
        with self.assertRaisesRegex(ValueError, "không có hóa đơn reviewed"):
            self.load(self.dataset, "valid", lambda message: None)

    def test_missing_both_paths_explains_upload(self):
        with self.assertRaisesRegex(FileNotFoundError, "upload groundtruth/reviewed/valid.jsonl"):
            self.load(self.dataset, "valid", lambda message: None)


if __name__ == "__main__":
    unittest.main()
