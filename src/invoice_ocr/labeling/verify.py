"""Extract an independent answer from the original PDF, then compare labels."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import sys
import time

from invoice_ocr.config import ROOT
from invoice_ocr.labeling.gemini import extract_one, pdf_source
from invoice_ocr.storage.files import loads_exact, write_json, images_digest
from invoice_ocr.labeling.comparison import annotation_hash, compare_annotations, has_content
from invoice_ocr.contracts.invoice import validate_invoice
from invoice_ocr.labeling.review import Store, revision


def read_settings(path):
    allowed = ('CHECK_API_KEY','CHECK_MODEL','CHECK_MAX_TOKENS','CHECK_REQUEST_INTERVAL')
    settings = {}
    if path.is_file():
        for line in path.read_text(encoding='utf-8-sig').splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            key, separator, value = line.partition('=')
            if not separator or key.strip() not in (*allowed, 'GEMINI_API_KEY','GEMINI_MODEL','GEMINI_REQUEST_INTERVAL'):
                raise ValueError('Unknown setting in checker env; use configs/labeling.env.example')
            settings[key.strip()] = value.strip().strip('"'+chr(39))
    settings.update({key:os.environ[key] for key in allowed if key in os.environ})
    return settings


def extract_check(record, model, prompt, max_tokens, settings):
    # Only the original PDF and shared prompt are sent, never model A's answer.
    from google import genai
    from google.genai import types
    with genai.Client(api_key=settings['CHECK_API_KEY']) as client:
        result = extract_one(client, types, record, model, prompt, max_tokens)
        if result.get('retryable'):
            result['error'] += '; temporary provider limit, retry later'
        return result


def apply_judgment(store, original, result):
    if original.get('review_status') == 'reviewed' or 'editor_draft' in original:
        raise ValueError('Do not replace reviewed labels or unfinished drafts')
    first_model = (original.get('review') or {}).get('model')
    if first_model == result.get('model'):
        raise ValueError('Checker must use a different model from the original labeler')
    pdf_source(original)
    if (result.get('input_type') != 'pdf' or result.get('source') != original['source']
            or result.get('source_sha256') != original['source_sha256']):
        raise ValueError('Checker result does not match the original PDF')
    if ((original.get('review') or {}).get('input_type') != 'pdf'
            or (original.get('review') or {}).get('source_sha256') != original['source_sha256']
            or (original.get('review') or {}).get('prompt_sha256') != result['prompt_sha256']):
        raise ValueError('First label must use the same original PDF and shared prompt')
    expected_images = images_digest(original['images'])
    if result.get('images_sha256') != expected_images or images_digest(result.get('images',[])) != expected_images:
        raise ValueError('Independent checker result does not match the invoice images/order')
    incoming = copy.deepcopy(original)
    incoming['_disk_revision'] = revision(original)
    checked_at = datetime.now(timezone.utc).isoformat()
    verification = {'status':'error', 'model':result['model'], 'checked_at':checked_at,
                    'annotation_sha256':annotation_hash(original['annotation']),
                    'images_sha256':expected_images, 'prompt_sha256':result['prompt_sha256'],
                    'input_type':'pdf', 'source_sha256':original['source_sha256']}
    if result.get('status') == 'ok':
        candidate = result['extraction']
        validate_invoice(candidate)
        differences = compare_annotations(original['annotation'], candidate)
        verification.update(status='disagreed' if differences else 'agreed',
                            candidate=candidate, differences=differences)
    else:
        verification['error'] = result.get('error','Independent checker extraction failed')
    agreed = verification['status'] == 'agreed'
    incoming['verification'] = verification
    incoming['review_status'] = 'reviewed' if agreed else 'needs_review'
    incoming['review'] = {**(original.get('review') or {}),
                          'reviewed_at':checked_at if agreed else None,
                          'action':'automatic_agreement' if agreed else 'checker_disagreement' if verification['status']=='disagreed' else 'judge_error'}
    # Store checks the original revision again after the API call. A human save wins.
    store.update(incoming)
    return verification['status']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=('train','valid','test','all'), default='all')
    parser.add_argument('--groundtruth', type=Path, default=ROOT/'groundtruth')
    parser.add_argument('--env-file', type=Path, default=ROOT/'.env.labeling')
    parser.add_argument('--model', help='Independent checker; otherwise CHECK_MODEL in .env.labeling')
    parser.add_argument('--document-id')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--max-tokens', type=int, help='Output budget; defaults to CHECK_MAX_TOKENS or 8192')
    parser.add_argument('--request-interval', type=float, help='Seconds between requests; defaults to CHECK_REQUEST_INTERVAL or 5')
    cache_root = ROOT/'judge/outputs' if (ROOT/'judge/outputs').is_dir() else ROOT/'outputs/labeling/verify'
    parser.add_argument('--output', type=Path, default=cache_root)
    parser.add_argument('--plan', action='store_true', help='List work without an API call')
    parser.add_argument('--apply', action='store_true', help='Apply comparison and auto-review full matches')
    parser.add_argument('--overwrite', action='store_true', help='Recheck pending/disagreed labels; never overwrite reviewed labels')
    parser.add_argument('--reset', action='store_true', help='Reset selected splits to pending independent verification; no API calls')
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1 or args.max_tokens is not None and args.max_tokens < 1:
        parser.error('limit and max-tokens must be positive')
    if args.reset and (args.document_id or args.limit or args.plan or args.apply or args.overwrite):
        parser.error('--reset applies to entire selected splits; use it alone with --split')
    splits = ('train','valid','test') if args.split == 'all' else (args.split,)
    store = Store(args.groundtruth)
    if args.reset:
        for split, count in store.reset(splits).items():
            print(split, count, 'labels reset to pending verification; reviewed subset emptied', flush=True)
        return 0
    settings = read_settings(args.env_file)
    try:
        args.max_tokens = args.max_tokens if args.max_tokens is not None else int(settings.get('CHECK_MAX_TOKENS') or 8192)
        interval = args.request_interval if args.request_interval is not None else float(settings.get('CHECK_REQUEST_INTERVAL') or 5)
        if args.max_tokens < 1 or not 0 <= interval < float('inf'):
            raise ValueError()
    except ValueError:
        parser.error('Output tokens must be positive; request interval must be finite and nonnegative')
    model = args.model or settings.get('CHECK_MODEL') or 'gemini-3.5-flash-lite'
    with store.lock, store.disk_lock():
        records = [r for split in splits for r in store.read(split)
                   if r.get('review_status') != 'reviewed' and 'editor_draft' not in r
                   and has_content(r['annotation'])
                   and (not args.document_id or r['document_id']==args.document_id)
                   and (args.overwrite or r.get('verification',{}).get('status','pending') in ('pending','error'))]
    if args.limit:
        records = records[:args.limit]
    print(f'{len(records)} invoices for independent checker; model={model or "not configured"}', flush=True)
    if args.plan:
        for record in records:
            print(record['split'], record['document_id'], 'PDF', record['source'], flush=True)
        return 0
    if not records:
        return 0
    if not model:
        parser.error('Set CHECK_MODEL in .env.labeling or pass --model')
    if not settings.get('CHECK_API_KEY'):
        parser.error('Set a Google Gemini CHECK_API_KEY in .env.labeling')
    if any((r.get('review') or {}).get('model') == model for r in records):
        parser.error('Checker must use a different model from the original labeler')
    prompt = (ROOT/'prompts/invoice.txt').read_text(encoding='utf-8')
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
    failures = 0
    last_request_finished = None
    for record in records:
        try:
            path = args.output/record['split']/(record['document_id']+'.json')
            pdf_source(record)
            result = None
            if path.is_file() and not args.overwrite:
                cached = loads_exact(path.read_text(encoding='utf-8-sig'))
                if (cached.get('status')=='ok' and cached.get('model')==model
                        and cached.get('input_type')=='pdf'
                        and cached.get('source')==record['source']
                        and cached.get('source_sha256')==record['source_sha256']
                        and cached.get('prompt_sha256')==prompt_hash
                        and cached.get('images_sha256')==images_digest(record['images'])):
                    validate_invoice(cached['extraction'])
                    result = cached
            if result is None:
                if last_request_finished is not None:
                    delay = interval-(time.monotonic()-last_request_finished)
                    if delay > 0:
                        print(f'Waiting {delay:.1f}s before the next request...',flush=True)
                        time.sleep(delay)
                result = extract_check(record, model, prompt, args.max_tokens, settings)
                if result.get('request_sent'):
                    last_request_finished = time.monotonic()
                write_json(path, result)
            if args.apply:
                state = apply_judgment(store, record, result)
            else:
                state = ('disagreed' if compare_annotations(record['annotation'],result['extraction']) else 'agreed') if result.get('status')=='ok' else 'error'
            print(record['split'], record['document_id'], state, 'applied' if args.apply else 'saved only', flush=True)
            if state == 'error':
                print('ERROR:', result.get('error','Independent checker failed'), flush=True)
            failures += state == 'error'
            if state == 'error':
                message = result.get('error','').lower()
                output_cap = 'output tokens per minute' in message and ('request too large' in message or 'expected output tokens exceed' in message)
                if result.get('retryable') or output_cap:
                    print('Stopped to avoid repeated quota errors. Remaining invoices were not changed. Check account limits or retry later.',flush=True)
                    break
        except (ValueError, OSError) as error:
            failures += 1
            print(record['document_id'], 'ERROR:', error, flush=True)
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
