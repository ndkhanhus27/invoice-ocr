"""GPU setup and vision-language model adapter, independent of invoice business rules."""
from __future__ import annotations

import importlib.metadata
import json
import re
import shutil
import subprocess
import sys
import time


def gpu_hardware():
    binary = shutil.which("nvidia-smi")
    if not binary:
        raise RuntimeError("No NVIDIA GPU detected (nvidia-smi missing). Select a Colab GPU runtime or check the GPU/driver on your local or SSH host. Installing torch cannot add a GPU to a CPU runtime.")
    result = subprocess.run([binary, "--query-gpu=name,driver_version", "--format=csv,noheader"],
                            capture_output=True, text=True, check=False)
    if result.returncode or not result.stdout.strip():
        raise RuntimeError("No accessible NVIDIA GPU. Check GPU allocation and the NVIDIA driver on this machine. " + result.stderr.strip())
    print("NVIDIA GPU / driver:\n" + result.stdout.strip(), flush=True)
    return result.stdout.strip()

def probe_torch():
    # New Python process avoids stale torch imports in a notebook kernel.
    snippet = "import json,torch; print(json.dumps({'version':torch.__version__,'cuda':torch.version.cuda,'available':torch.cuda.is_available()}))"
    result = subprocess.run([sys.executable, "-c", snippet], capture_output=True, text=True, check=False)
    if result.returncode:
        return {"version": None, "cuda": None, "available": False, "error": result.stderr.strip()}
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        raise RuntimeError("Cannot parse PyTorch GPU probe: " + result.stdout)

def setup_gpu(args):
    if not args.torch_index_url.startswith("https://download.pytorch.org/whl/cu"):
        raise ValueError("Use an official CUDA wheel index: https://download.pytorch.org/whl/cu...")
    gpu_hardware()  # Must pass before downloads/package changes.
    before = probe_torch()
    print("PyTorch before:", json.dumps(before), flush=True)
    if not before["available"]:
        if args.check_only:
            raise RuntimeError("GPU hardware exists but PyTorch CUDA is unavailable. Run invoice.py setup without --check-only to repair a CPU/missing torch build.")
        if before["cuda"] is not None and before["version"]:
            raise RuntimeError("PyTorch already has CUDA support but cannot use the GPU. Check the NVIDIA driver/runtime and GPU visibility; refusing to replace a CUDA build blindly.")
        print("Installing matching CUDA torch/torchvision wheels...", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-cache-dir",
                        f"torch=={args.torch_version}", f"torchvision=={args.vision_version}",
                        "--index-url", args.torch_index_url], check=True)
        after = probe_torch()
        if not after["available"]:
            raise RuntimeError("CUDA still unavailable after wheel installation. Inspect GPU/driver compatibility: " + json.dumps(after))
    if not args.check_only:
        if not args.requirements.is_file():
            raise FileNotFoundError(args.requirements)
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(args.requirements)], check=True)
        if not probe_torch()["available"]:
            raise RuntimeError("CUDA became unavailable after dependency installation; review pip output.")
    print('CUDA ready. Continue with invoice-ocr gpu doctor / gpu download / extract.', flush=True)
    return 0

def doctor(_args):
    versions = {}
    for name in ("torch", "torchvision", "transformers", "accelerate", "bitsandbytes", "PyMuPDF"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    print(json.dumps({"python": sys.version, "packages": versions}, indent=2))
    binary = shutil.which("nvidia-smi")
    if binary:
        hardware = subprocess.run([binary, "--query-gpu=name,driver_version", "--format=csv,noheader"],
                                  capture_output=True, text=True, check=False)
        print("NVIDIA hardware:", hardware.stdout.strip() or hardware.stderr.strip())
    else:
        print("NVIDIA hardware: nvidia-smi not found")
    try:
        import torch
    except ImportError:
        raise RuntimeError("PyTorch is missing. Run: python -m invoice_ocr gpu setup") from None
    print("CUDA runtime:", torch.version.cuda, "available:", torch.cuda.is_available())
    if not torch.cuda.is_available():
        if torch.version.cuda is None:
            raise RuntimeError("CPU-only PyTorch is installed. Run python -m invoice_ocr gpu setup on a Colab GPU runtime; setup checks GPU hardware before installing CUDA wheels.")
        raise RuntimeError("CUDA-enabled PyTorch cannot access the GPU. Check GPU allocation, NVIDIA driver and runtime visibility; run python -m invoice_ocr gpu setup --check-only.")
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        print(index, props.name, round(props.total_memory / 1024**3, 2), "GiB")

def download(args):
    from huggingface_hub import snapshot_download
    folder = snapshot_download(
        repo_id=args.model_id, revision=args.revision, local_dir=str(args.output),
        allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt"],
    )
    print(folder)

def load_model(args):
    import torch
    from transformers import AutoConfig, AutoProcessor, AutoModelForImageTextToText, MllamaForConditionalGeneration
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required. Run the doctor command first.")
    if args.device != "auto" and not re.fullmatch(r"cuda:\d+", args.device):
        raise ValueError("--device must be auto or cuda:N")
    common = {"local_files_only": args.offline, "revision": args.revision}
    config = AutoConfig.from_pretrained(args.model, **common)
    quant = getattr(config, "quantization_config", None)
    if not isinstance(quant, dict) or not quant.get("load_in_4bit") or quant.get("quant_method") != "bitsandbytes":
        raise ValueError("Use a pre-quantized BitsAndBytes 4-bit checkpoint")
    # Override BF16 compute on T4; preserve skip_modules and prequantized metadata.
    quant["bnb_4bit_compute_dtype"] = "float16"
    model_class = MllamaForConditionalGeneration if config.model_type == "mllama" else AutoModelForImageTextToText
    model = model_class.from_pretrained(
        args.model, config=config, dtype=torch.float16,
        device_map="auto" if args.device == "auto" else {"": args.device},
        attn_implementation="sdpa", low_cpu_mem_usage=True,
        **common,
    ).eval()
    if not getattr(model, "is_loaded_in_4bit", False):
        raise RuntimeError("Model was not loaded in 4-bit")
    processor = AutoProcessor.from_pretrained(args.model, **common)
    print("Model loaded, 4-bit=True, map=", model.hf_device_map, flush=True)
    return model, processor


class TransformersBackend:
    """Keep tensor/model details outside the shared extraction pipeline."""
    def __init__(self, args, *, model_loader=load_model):
        self.args = args
        self.model_loader = model_loader
        self.loaded = None

    def check_model(self):
        if self.loaded is None:
            self.loaded = self.model_loader(self.args)
        return {'digest': self.args.revision}

    def generate(self, frames, prompt):
        import torch
        model, processor = self.loaded
        content = []
        for index, frame in enumerate(frames, 1):
            content.extend([{'type': 'text', 'text': f'Invoice image {index}/{len(frames)}:'},
                            {'type': 'image', 'image': frame}])
        content.append({'type': 'text', 'text': prompt})
        inputs = output = None
        try:
            inputs = processor.apply_chat_template([{'role': 'user', 'content': content}], tokenize=True,
                add_generation_prompt=True, return_dict=True, return_tensors='pt').to(model.device)
            if getattr(getattr(model, 'config', None), 'model_type', None) == 'mllama':
                inputs = inputs.to(dtype=model.dtype)
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=self.args.max_new_tokens, do_sample=False)
            torch.cuda.synchronize()
            inference_seconds = round(time.perf_counter() - started, 3)
            ids = output[0, inputs['input_ids'].shape[1]:]
            text = processor.decode(ids, skip_special_tokens=True)
            return {'done': True, 'done_reason': 'length' if len(ids) >= self.args.max_new_tokens else 'stop',
                    'eval_count': len(ids), 'message': {'content': text}, 'inference_seconds': inference_seconds}
        finally:
            del inputs, output
            torch.cuda.empty_cache()
