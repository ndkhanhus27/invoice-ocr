"""Original-weight loader checks without CUDA/model downloads."""
import contextlib
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from invoice_ocr.extraction.backends import transformers as backend


class OriginalBf16Tests(unittest.TestCase):
    def setUp(self):
        self.args = SimpleNamespace(model="original-model", revision="a" * 40, offline=True,
                                    device="cuda:0", weights_format="bf16", vision_soft_tokens=1120)
        self.config = SimpleNamespace(model_type="gemma4", quantization_config=None)
        self.model = SimpleNamespace(dtype="bf16", hf_device_map={"": "cuda:0"},
                                     is_loaded_in_4bit=False, is_loaded_in_8bit=False)
        self.model.eval = Mock(return_value=self.model)
        self.processor = SimpleNamespace(image_processor=SimpleNamespace())
        self.loader = Mock(from_pretrained=Mock(return_value=self.model))
        self.modules = {
            "torch": SimpleNamespace(bfloat16="bf16", float16="fp16", cuda=SimpleNamespace(
                is_available=lambda: True, is_bf16_supported=lambda: True,
                device=lambda value: contextlib.nullcontext())),
            "transformers": SimpleNamespace(AutoConfig=Mock(from_pretrained=Mock(return_value=self.config)),
                AutoProcessor=Mock(from_pretrained=Mock(return_value=self.processor)),
                AutoModelForImageTextToText=self.loader, MllamaForConditionalGeneration=self.loader,
                Gemma4ForConditionalGeneration=self.loader),
        }

    def load(self):
        with patch.dict("sys.modules", self.modules):
            return backend.load_model(self.args)

    def test_original_bf16_is_not_quantized_or_cpu_offloaded(self):
        self.assertEqual(self.load(), (self.model, self.processor))
        options = self.loader.from_pretrained.call_args.kwargs
        self.assertEqual(options["dtype"], "bf16")
        self.assertEqual(options["device_map"], {"": "cuda:0"})
        self.assertTrue(options["low_cpu_mem_usage"])
        self.assertTrue(options["local_files_only"])
        self.assertNotIn("quantization_config", options)
        self.assertEqual(self.processor.image_processor.max_soft_tokens, 1120)

    def test_quantized_checkpoint_is_rejected_for_original_mode(self):
        self.config.quantization_config = {"load_in_4bit": True}
        with self.assertRaisesRegex(ValueError, "original unquantized"):
            self.load()
        self.loader.from_pretrained.assert_not_called()

    def test_auto_offload_is_rejected(self):
        self.args.device = "auto"
        with self.assertRaisesRegex(ValueError, "explicit cuda"):
            self.load()
        self.loader.from_pretrained.assert_not_called()

    def test_unexpected_cpu_placement_is_rejected(self):
        self.model.hf_device_map = {"some_layer": "cpu"}
        with self.assertRaisesRegex(RuntimeError, "offload"):
            self.load()

    def test_gemma4_without_hf_device_map_loads_and_logs_actual_devices(self):
        del self.model.hf_device_map
        self.model.parameters = lambda: iter([SimpleNamespace(device="cuda:0")])
        self.assertEqual(self.load(), (self.model, self.processor))

    def test_missing_map_does_not_hide_cpu_parameters(self):
        del self.model.hf_device_map
        self.model.parameters = lambda: iter([SimpleNamespace(device="cuda:0"), SimpleNamespace(device="cpu")])
        with self.assertRaisesRegex(RuntimeError, "offload"):
            self.load()

    def test_legacy_bnb4_loading_still_uses_fp16_compute(self):
        self.args.weights_format = "bnb4"
        self.config.model_type = "mllama"
        self.config.quantization_config = {"quant_method": "bitsandbytes", "load_in_4bit": True}
        self.model.is_loaded_in_4bit = True
        self.load()
        self.assertEqual(self.loader.from_pretrained.call_args.kwargs["dtype"], "fp16")
        self.assertEqual(self.config.quantization_config["bnb_4bit_compute_dtype"], "float16")


class Gemma4GenerationTests(unittest.TestCase):
    def fake(self, input_length=12):
        class Inputs(dict):
            def to(self, *args, **kwargs):
                return self
        class Output:
            def __getitem__(self, value):
                return [10, 11, 12]
        processor = SimpleNamespace(
            apply_chat_template=Mock(return_value=Inputs(input_ids=SimpleNamespace(shape=(1, input_length)))),
            decode=lambda ids, skip_special_tokens: ('{"items":[]}' if skip_special_tokens else
                '<|channel>thought\n<channel|>{"items":[]}<turn|>'),
        )
        model = SimpleNamespace(device="cuda:0", dtype="bf16", config=SimpleNamespace(model_type="gemma4"),
                                generate=Mock(return_value=Output()))
        torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None, empty_cache=lambda: None),
                                inference_mode=contextlib.nullcontext)
        client = backend.TransformersBackend(SimpleNamespace(max_new_tokens=4096, max_input_tokens=8192))
        client.loaded = model, processor
        return client, processor, model, torch

    def test_gemma4_uses_all_images_and_disables_thinking(self):
        client, processor, model, torch = self.fake()
        with patch.dict("sys.modules", {"torch": torch}):
            response = client.generate(["page1", "page2"], "extract invoice")
        call = processor.apply_chat_template.call_args
        self.assertFalse(call.kwargs["enable_thinking"])
        self.assertEqual(sum(part["type"] == "image" for part in call.args[0][0]["content"]), 2)
        self.assertEqual(response["message"]["content"], '{"items":[]}')
        self.assertFalse(model.generate.call_args.kwargs["do_sample"])

    def test_input_limit_fails_without_dropping_invoice_pages(self):
        client, _, model, torch = self.fake(input_length=9000)
        with patch.dict("sys.modules", {"torch": torch}), self.assertRaisesRegex(ValueError, "no pages were dropped"):
            client.generate(["page1"], "extract")
        model.generate.assert_not_called()
