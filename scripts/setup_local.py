"""Prepare local config/services and launch the upload UI without CUDA dependencies."""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys

# This operational script can bootstrap a checkout before editable installation.
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from invoice_ocr.config import ROOT, parse_args, read_env
from invoice_ocr.extraction.backends.ollama import doctor
from invoice_ocr.storage.postgres import initialize_database, postgres_connection


def update_env(path, updates):
    """Update named settings, preserving other settings and never printing secrets."""
    lines = path.read_text(encoding='utf-8-sig').splitlines()
    written = set()
    for index, line in enumerate(lines):
        key = line.partition('=')[0].strip()
        if key in updates:
            lines[index] = key + '=' + updates[key]
            written.add(key)
    lines.extend(key + '=' + value for key, value in updates.items() if key not in written)
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def ensure_config(path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / 'configs/ollama.env.example', path)
        print('Created local config from configs/ollama.env.example (no credentials printed).')
    if 'INVOICE_BACKEND' not in read_env(path) and 'INVOICE_BACKEND' not in os.environ:
        update_env(path, {'INVOICE_BACKEND': 'ollama'})
        print('Enabled Ollama in existing config; other settings preserved.')


def ollama_executable():
    binary = shutil.which('ollama')
    if not binary and os.environ.get('LOCALAPPDATA'):
        candidate = Path(os.environ['LOCALAPPDATA']) / 'Programs/Ollama/ollama.exe'
        if candidate.is_file():
            binary = str(candidate)
    if not binary:
        raise RuntimeError('Install Ollama from https://ollama.com/download/windows, open it, then rerun this script.')
    return binary


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-file', type=Path, default=ROOT / '.env')
    parser.add_argument('--pull-model', action='store_true')
    parser.add_argument('--start-database', action='store_true')
    parser.add_argument('--initialize-database', action='store_true')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--port', type=int, default=7860)
    options = parser.parse_args()
    env_file = options.env_file.resolve()
    ensure_config(env_file)
    missing = [name for name in ('gradio', 'PIL', 'psycopg') if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError('Missing packages: ' + ', '.join(missing) + '. Run: python -m pip install -r requirements/local.txt')
    if options.start_database:
        if not shutil.which('docker'):
            raise RuntimeError('Install/start Docker Desktop or use an existing PostgreSQL connection string.')
        probe = subprocess.run(['docker', 'info', '--format', '{{.ServerVersion}}'], capture_output=True, timeout=30)
        if probe.returncode:
            raise RuntimeError('Docker engine is not running. Open Docker Desktop and wait for it to be ready.')
        values = read_env(env_file)
        values.update({k: v for k, v in os.environ.items() if k.startswith('INVOICE_')})
        password = values.get('INVOICE_POSTGRES_PASSWORD') or secrets.token_urlsafe(24)
        port = int(values.get('INVOICE_POSTGRES_PORT') or 5432)
        if not 1 <= port <= 65535:
            raise ValueError('INVOICE_POSTGRES_PORT must be between 1 and 65535.')
        updates = {}
        if not values.get('INVOICE_POSTGRES_PASSWORD'):
            updates['INVOICE_POSTGRES_PASSWORD'] = password
        if not values.get('INVOICE_DATABASE_URL'):
            from urllib.parse import quote
            updates['INVOICE_DATABASE_URL'] = 'postgresql://invoice:' + quote(password, safe='') + f'@127.0.0.1:{port}/invoice_ocr'
        if updates:
            update_env(env_file, updates)
        compose_env = dict(os.environ, INVOICE_POSTGRES_PASSWORD=password, INVOICE_POSTGRES_PORT=str(port))
        # Suppress compose config output, which can contain expanded passwords.
        result = subprocess.run(['docker', 'compose', '--project-name', 'invoice-ocr', '--env-file', str(env_file), '-f', str(ROOT/'deploy/compose.yaml'),
                                 'up', '-d', '--wait', 'postgres'], cwd=ROOT, env=compose_env, capture_output=True)
        if result.returncode:
            raise RuntimeError(f'Could not start PostgreSQL container. Check Docker Desktop, image downloads and local port {port}.')
        print('PostgreSQL container ready; credentials stored only in local config.')
    args = parse_args(['--env-file', str(env_file), 'extract'])
    if args.backend == 'ollama':
        if options.pull_model:
            result = subprocess.run([ollama_executable(), 'pull', args.ollama_model],
                                    env=dict(os.environ, OLLAMA_HOST=args.ollama_url), check=False)
            if result.returncode:
                raise RuntimeError('Model download failed. Start Ollama and check server URL/network access.')
        doctor(args)
    else:
        from invoice_ocr.extraction.backends.transformers import doctor as gpu_doctor
        gpu_doctor(args)
    if options.initialize_database:
        initialize_database(args)
    if args.database_url:
        with postgres_connection(args) as connection:
            row = connection.execute("SELECT to_regclass('invoice_ocr.invoices')").fetchone()
            if row[0] is None:
                raise RuntimeError('Database schema is missing. Rerun with -InitializeDatabase or run invoice-ocr db init.')
        print('PostgreSQL connection and invoice schema ready.')
    else:
        print('No database configured. Extraction works; database save is disabled.')
    if options.check_only:
        return 0
    from app.service import UploadService
    from app.ui import build_ui
    build_ui(UploadService(args)).queue(max_size=8).launch(
        server_name='127.0.0.1', server_port=options.port, share=False, max_file_size='20mb')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(run())
    except Exception as error:
        # DB errors are sanitized by postgres_connection; don't print raw driver exceptions.
        print(f'ERROR: {error}', file=sys.stderr)
        sys.exit(1)
