"""Exercise notebook subprocess logging without marimo, network, data or model."""
import ast
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


def load_stream_command():
    path = Path(__file__).resolve().parents[1] / "notebooks/molab_gemma4.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
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


if __name__ == "__main__":
    unittest.main()
