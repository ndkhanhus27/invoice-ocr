"""Single command surface; import optional model/UI SDKs only when needed."""
from __future__ import annotations

import importlib
import argparse
import json
import sys

from .config import parse_args
from .storage.files import json_records
from .storage.postgres import postgres_connection, store_report


def load_database(args, *, connection_factory=None, writer=None):
    connection_factory = connection_factory or postgres_connection
    writer = writer or store_report
    _, records = json_records(args.input, normalized=True)
    counts = {'loaded': 0, 'already_loaded': 0, 'staged_for_review': 0, 'errors': 0}
    with connection_factory(args) as connection:
        for path, report in records:
            try:
                outcome = writer(connection, report)
                counts[outcome] += 1
                print(outcome, path.name, flush=True)
            except Exception as error:
                counts['errors'] += 1
                print(f'DB write failed for {path.name} ({type(error).__name__}); transaction rolled back', flush=True)
    print(json.dumps(counts))
    return 1 if counts['errors'] else 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument('--env-file')
    selected, forwarded = bootstrap.parse_known_args(argv)
    if forwarded and forwarded[0] in ('evaluate', 'ui'):
        module = 'invoice_ocr.evaluation.runner' if forwarded[0] == 'evaluate' else 'app.ui'
        arguments = forwarded[1:]
        if selected.env_file and forwarded[0] == 'ui':
            arguments += ['--env-file', selected.env_file]
        result = importlib.import_module(module).main(arguments)
        return result or 0
    if len(forwarded) >= 2 and forwarded[0] == 'label' and forwarded[1] in ('prepare', 'review', 'generate', 'verify', 'export'):
        module = {'generate': 'gemini'}.get(forwarded[1], forwarded[1])
        arguments = forwarded[2:]
        if selected.env_file and forwarded[1] in ('generate', 'verify'):
            arguments += ['--env-file', selected.env_file]
        result = importlib.import_module('invoice_ocr.labeling.' + module).main(arguments)
        return result or 0
    args = parse_args(argv)
    if args.command in ('extract', 'plan'):
        from .extraction.pipeline import extract, jobs
        if args.command == 'extract':
            return extract(args)
        for source, target in jobs(args):
            print(f'{source} -> {target}')
        return 0
    if args.command == 'db':
        if args.db_action == 'load':
            return load_database(args)
        from .storage.postgres import initialize_database
        return initialize_database(args)
    if args.command == 'gpu':
        from .extraction.backends import transformers
        return {'setup': transformers.setup_gpu, 'doctor': transformers.doctor,
                'download': transformers.download}[args.gpu_action](args) or 0
    if args.command == 'ollama':
        from .extraction.backends.ollama import doctor
        return doctor(args)
    return 0


def entrypoint(argv=None):
    try:
        raise SystemExit(main(argv))
    except Exception as error:
        print(f'ERROR: {type(error).__name__}: {error}', file=sys.stderr)
        raise SystemExit(1) from None
