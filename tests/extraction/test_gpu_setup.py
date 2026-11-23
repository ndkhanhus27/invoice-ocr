import types
import unittest
from unittest.mock import patch
from invoice_ocr.config import ROOT
from invoice_ocr.extraction.backends import transformers as gpu


class SharedGpuSetupTests(unittest.TestCase):
    def args(self, check_only=False):
        return types.SimpleNamespace(check_only=check_only, requirements=ROOT / "requirements/gpu.txt",
            torch_index_url="https://download.pytorch.org/whl/cu126", torch_version="2.14.1", vision_version="0.29.1")

    def test_no_gpu_stops_before_install(self):
        with patch.object(gpu, "gpu_hardware", side_effect=RuntimeError("No GPU")), patch.object(gpu.subprocess, "run") as install:
            with self.assertRaisesRegex(RuntimeError, "No GPU"):
                gpu.setup_gpu(self.args())
            install.assert_not_called()

    def test_healthy_cuda_check_only_does_not_install(self):
        healthy = {"version": "2.14.1+cu126", "cuda": "12.6", "available": True}
        with patch.object(gpu, "gpu_hardware"), patch.object(gpu, "probe_torch", return_value=healthy), patch.object(gpu.subprocess, "run") as install:
            self.assertEqual(gpu.setup_gpu(self.args(check_only=True)), 0)
            install.assert_not_called()

    def test_cpu_torch_with_gpu_repairs_then_installs_shared_requirements(self):
        cpu = {"version": "2.14.1+cpu", "cuda": None, "available": False}
        healthy = {"version": "2.14.1+cu126", "cuda": "12.6", "available": True}
        with patch.object(gpu, "gpu_hardware"), patch.object(gpu, "probe_torch", side_effect=[cpu, healthy, healthy]), patch.object(gpu.subprocess, "run") as install:
            self.assertEqual(gpu.setup_gpu(self.args()), 0)
            self.assertEqual(install.call_count, 2)
            self.assertIn("torch==2.14.1", install.call_args_list[0].args[0])
            self.assertIn("torchvision==0.29.1", install.call_args_list[0].args[0])
            self.assertIn(str(ROOT / "requirements/gpu.txt"), install.call_args_list[1].args[0])
