"""Shared image discovery, identity, validation and output for every model backend."""
from __future__ import annotations

import hashlib
import time

from ..storage.files import documents, file_sha256, image_paths, images_digest, load_images, write_json
from ..contracts.invoice import parse_invoice
from .backends.transformers import load_model, TransformersBackend


def jobs(args):
    _, invoices = documents(args.input)
    if args.input.is_dir() and args.output.resolve().is_relative_to(args.input.resolve()):
        raise ValueError('Output must be outside the image input directory')
    if args.limit:
        invoices = invoices[:args.limit]
    result = [(path, args.output / ((path.stem if path.is_file() else path.name) + '.json')) for path in invoices]
    for _, target in result:
        if target.exists() and not args.overwrite:
            raise ValueError(f'Output exists: {target}; choose a new run or --overwrite')
    return result


def extract(args, *, model_loader=None, image_loader=None, client_factory=None):
    work = jobs(args)
    prompt = args.prompt.read_text(encoding='utf-8')
    backend_name = getattr(args, 'backend', 'transformers')
    if backend_name == 'ollama':
        from .backends.ollama import OllamaClient
        backend = (client_factory or OllamaClient)(args)
    elif backend_name == 'transformers':
        backend = TransformersBackend(args, model_loader=model_loader or load_model)
    else:
        raise ValueError('Unknown extraction backend: ' + str(backend_name))
    identity = None
    failures = 0
    for document, target in work:
        record = {'label_format': 'invoice-v3', 'document_id': document.stem if document.is_file() else document.name,
                  'source': str(document.resolve()), 'backend': backend_name,
                  'model': args.ollama_model if backend_name == 'ollama' else args.model,
                  'model_revision': None, 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                  'settings': {'max_edge': args.max_edge, 'max_new_tokens': args.max_new_tokens}, 'status': 'ok'}
        if backend_name == 'ollama':
            record['settings'].update(device=args.ollama_device, num_ctx=args.ollama_num_ctx,
                                      num_threads=args.ollama_num_threads, temperature=0, seed=42)
        frames = []
        try:
            record['images'] = [{'file': path.name, 'sha256': file_sha256(path)} for path in image_paths(document)]
            record['images_sha256'] = images_digest(record['images'])
            if identity is None:
                identity = backend.check_model()
            record['model_revision'] = identity.get('digest')
            frames = (image_loader or load_images)(document, args.max_edge)
            if len(frames) != len(record['images']):
                raise ValueError('All invoice images must be provided together')
            started = time.perf_counter()
            response = backend.generate(frames, prompt)
            record['inference_seconds'] = response.get('inference_seconds', round(time.perf_counter() - started, 3))
            if backend_name == 'ollama':
                record['ollama_metrics'] = {key: response[key] for key in (
                    'total_duration', 'load_duration', 'prompt_eval_count', 'prompt_eval_duration',
                    'eval_count', 'eval_duration', 'done_reason') if key in response}
            record['generated_tokens'] = response.get('eval_count')
            record['raw_text'] = response.get('message', {}).get('content', '')
            if not response.get('done') or response.get('done_reason') == 'length':
                raise ValueError('Incomplete model output: token limit reached; increase max_new_tokens/context and review')
            record['extraction'] = parse_invoice(record['raw_text'])
        except Exception as error:
            record.update(status='error', error=f'{type(error).__name__}: {error}')
            failures += 1
        finally:
            for frame in frames:
                frame.close()
        write_json(target, record)
        print(record['status'], target, flush=True)
    print(f'Finished {len(work)} invoices; {failures} errors', flush=True)
    return 1 if failures else 0
