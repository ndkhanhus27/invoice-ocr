"""Configuration and CLI argument parsing. No inference or persistence."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re

MODEL_ID = "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit"

ROOT = Path(__file__).resolve().parents[2]

def read_env(path):
    """Read simple KEY=VALUE data, never execute shell code or expand variables."""
    values = {}
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not separator or not re.fullmatch(r"INVOICE_[A-Z0-9_]+", key):
            raise ValueError(f"Invalid config line {number} in {path}; expected INVOICE_KEY=value")
        if value.startswith(('"', "'")):
            if len(value) < 2 or value[-1] != value[0]:
                raise ValueError(f"Unclosed quote at line {number} in {path}")
            value = value[1:-1]
        values[key] = value
    return values

def env_bool(value):
    value = value.strip().lower()
    if value not in {"true", "false", "1", "0", "yes", "no"}:
        raise ValueError("Boolean config values must be true/false, 1/0 or yes/no")
    return value in {"true", "1", "yes"}

def configured_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path

def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number

def parse_args(argv=None):
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--env-file", type=Path)
    selected, remaining = bootstrap.parse_known_args(argv)
    env_file = selected.env_file or ROOT / ".env"
    if selected.env_file and not env_file.is_file():
        raise ValueError(f"Config file does not exist: {env_file}")
    settings = read_env(env_file) if env_file.is_file() else {}
    settings.update({key: value for key, value in os.environ.items() if key.startswith("INVOICE_")})
    def setting(key, fallback):
        return settings.get("INVOICE_" + key) or fallback
    model_id = setting("MODEL_ID", MODEL_ID)
    model_dir = configured_path(setting("MODEL_DIR", "models/" + model_id.rsplit("/", 1)[-1]))
    parser = argparse.ArgumentParser(description="Shared invoice OCR CLI for local, SSH GPU hosts and Colab.")
    parser.add_argument("--env-file", type=Path, help="Config file, accepted before or after subcommand; default: project .env")
    sub = parser.add_subparsers(dest="command", required=True)
    gpu = sub.add_parser('gpu', help='Set up, inspect or download the Transformers GPU backend')
    gpu_sub = gpu.add_subparsers(dest='gpu_action', required=True)
    setup = gpu_sub.add_parser("setup", help="Check NVIDIA GPU and install CUDA dependencies")
    setup.add_argument("--check-only", action="store_true")
    setup.add_argument("--requirements", type=Path, default=ROOT / "requirements/gpu.txt")
    setup.add_argument("--torch-index-url", default=setting("TORCH_INDEX_URL", "https://download.pytorch.org/whl/cu126"))
    setup.add_argument("--torch-version", default=setting("TORCH_VERSION", "2.14.1"))
    setup.add_argument("--vision-version", default=setting("VISION_VERSION", "0.29.1"))
    gpu_sub.add_parser("doctor", help="Show package versions and CUDA GPUs")
    ollama = sub.add_parser('ollama', help='Inspect the Ollama server and selected vision model')
    ollama_sub = ollama.add_subparsers(dest='ollama_action', required=True)
    ollama_doctor = ollama_sub.add_parser('doctor', help='Check server, installed model and vision support')
    def ollama_options(command):
        command.add_argument('--ollama-url', default=setting('OLLAMA_URL', 'http://127.0.0.1:11434'))
        command.add_argument('--ollama-model', default=setting('OLLAMA_MODEL', 'qwen3-vl:2b'))
        command.add_argument('--ollama-device', choices=('cpu', 'auto'), default=setting('OLLAMA_DEVICE', 'cpu'))
        command.add_argument('--ollama-timeout', type=positive, default=setting('OLLAMA_TIMEOUT', '600'))
        command.add_argument('--ollama-num-ctx', type=positive, default=setting('OLLAMA_NUM_CTX', '8192'))
        command.add_argument('--ollama-num-threads', type=positive, default=setting('OLLAMA_NUM_THREADS', None))
        command.add_argument('--ollama-keep-alive', default=setting('OLLAMA_KEEP_ALIVE', '5m'))
    ollama_options(ollama_doctor)
    dl = gpu_sub.add_parser("download", help="Download prequantized 4-bit weights only")
    dl.add_argument("--output", type=Path, default=model_dir)
    dl.add_argument("--model-id", default=model_id, help="Hub repository to download")
    dl.add_argument("--revision", default=setting("MODEL_REVISION", "main"), help="Commit SHA to pin checkpoint")
    for name in ("plan", "extract"):
        run = sub.add_parser(name, help="List work without GPU/downloads" if name == "plan" else "Extract one JSON per image invoice")
        run.add_argument("--input", type=Path, default=configured_path(setting("INPUT_DIR", "data/valid")))
        run.add_argument("--output", type=Path, default=configured_path(setting("OUTPUT_DIR", "outputs/valid")))
        run.add_argument("--limit", type=positive, default=setting("LIMIT", None))
        run.add_argument("--overwrite", action="store_true")
        if name == "extract":
            run.add_argument('--backend', choices=('ollama', 'transformers'), default=setting('BACKEND', 'transformers'))
            ollama_options(run)
            run.add_argument("--model", default=str(model_dir), help="4-bit Hub ID or local model directory")
            run.add_argument("--revision", default=setting("MODEL_REVISION", "main"))
            run.add_argument("--offline", action=argparse.BooleanOptionalAction,
                             default=env_bool(setting("OFFLINE", "true")))
            run.add_argument("--device", default=setting("DEVICE", "cuda:0"), help="cuda:0, cuda:1, or auto")
            run.add_argument("--prompt", type=Path, default=configured_path(setting("PROMPT_PATH", "prompts/invoice.txt")))
            run.add_argument("--max-edge", type=positive, default=setting("MAX_EDGE", "1600"))
            run.add_argument("--max-new-tokens", type=positive, default=setting("MAX_NEW_TOKENS", "4096"))
    db = sub.add_parser('db', help='Initialize PostgreSQL or load saved predictions')
    db_sub = db.add_subparsers(dest='db_action', required=True)
    db_sub.add_parser('init', help='Create the invoice_ocr schema without dropping data')
    dbload = db_sub.add_parser('load', help='Load saved invoice-v3 predictions')
    dbload.add_argument('--input', type=Path, required=True)
    sub.add_parser('evaluate', help='Compare reviewed labels and predictions (use evaluate --help)', add_help=False)
    sub.add_parser('ui', help='Launch the Gradio upload interface (use ui --help)', add_help=False)
    label = sub.add_parser('label', help='Prepare, review, generate, verify or export labels')
    label_sub = label.add_subparsers(dest='label_action', required=True)
    for action in ('prepare', 'review', 'generate', 'verify', 'export'):
        label_sub.add_parser(action, help=f'{action.title()} invoice labels', add_help=False)
    args = parser.parse_args(remaining)
    args.env_file = env_file
    args.database_url = setting("DATABASE_URL", os.environ.get("DATABASE_URL"))
    return args
