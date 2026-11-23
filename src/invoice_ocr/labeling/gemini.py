"""Draft one invoice label from its original PDF using Gemini API. No local GPU."""
from __future__ import annotations
import argparse
import copy
import hashlib
import os
from pathlib import Path
import sys
import time

from invoice_ocr.config import ROOT
from invoice_ocr.storage.files import read_jsonl, loads_exact, write_json, file_sha256, images_digest
from invoice_ocr.contracts.invoice import invoice_json_schema as response_schema, parse_invoice, validate_invoice
from invoice_ocr.labeling.review import Store, revision

DEFAULT_MODEL = 'gemini-3.1-flash-lite'


def read_settings(path):
    values = {}
    if path.is_file():
        for line in path.read_text(encoding='utf-8-sig').splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            key, separator, value = line.partition('=')
            if not separator or key.strip() not in ('GEMINI_API_KEY','GEMINI_MODEL','GEMINI_REQUEST_INTERVAL','CHECK_API_KEY','CHECK_MODEL','CHECK_MAX_TOKENS','CHECK_REQUEST_INTERVAL'):
                raise ValueError('Use keys from configs/labeling.env.example in '+str(path))
            values[key.strip()] = value.strip().strip('\"\'')
    for key in ('GEMINI_API_KEY','GEMINI_MODEL','GEMINI_REQUEST_INTERVAL'):
        if key in os.environ:
            values[key] = os.environ[key]
    return values


def pdf_source(record):
    """Validate original PDF identity; never render or send the preview images."""
    path = (ROOT / record['source']).resolve()
    if not path.is_relative_to((ROOT / 'data_split').resolve()) or path.suffix.lower() != '.pdf':
        raise ValueError('Labeling requires an original PDF under data_split: '+record['source'])
    if not path.is_file() or file_sha256(path) != record['source_sha256']:
        raise ValueError('Original PDF missing or changed: '+record['source'])
    return path


def extract_one(client, types, record, model, prompt, max_tokens):
    pdf = pdf_source(record)
    images = record['images']  # Preview identity only; never included in the API request.
    result = {'label_format':'invoice-v3', 'document_id':record['document_id'],
              'images':images, 'images_sha256':images_digest(images),
              'input_type':'pdf', 'source':record['source'], 'source_sha256':record['source_sha256'],
              'model':model, 'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
              'status':'error'}
    uploaded = None
    started = time.perf_counter()
    try:
        if pdf.stat().st_size > 50 * 1024 * 1024:
            raise ValueError('PDF exceeds 50 MiB; no pages were dropped')
        if pdf.stat().st_size * 4 // 3 + len(prompt.encode()) < 15_000_000:
            document = types.Part.from_bytes(data=pdf.read_bytes(), mime_type='application/pdf')
        else:
            uploaded = client.files.upload(file=pdf, config={'mime_type':'application/pdf'})
            document = uploaded
        result['request_sent'] = True
        response = client.models.generate_content(model=model, contents=[prompt, document], config={
            'response_mime_type':'application/json', 'response_json_schema':response_schema(),
            'temperature':0, 'max_output_tokens':max_tokens})
        result['raw_text'] = response.text or ''
        candidates = response.candidates or []
        reason = str(candidates[0].finish_reason).split('.')[-1] if candidates else ''
        if reason != 'STOP':
            raise ValueError('Gemini did not finish normally: '+reason)
        result['extraction'] = parse_invoice(result['raw_text'])
        result['status'] = 'ok'
    except Exception as error:
        # Never save credentials or full provider request details.
        result['retryable'] = getattr(error, 'code', None) in (429, 503)
        result['error'] = str(error) if isinstance(error, ValueError) else type(error).__name__+'; check model access, quota, network or API key'
    finally:
        if uploaded is not None:
            try:
                client.files.delete(name=uploaded.name)
            except Exception:
                pass
        result['inference_seconds'] = round(time.perf_counter()-started, 3)
    return result


def apply_draft(store, original, result):
    """Optimistic revision check prevents overwriting a human save made during the API call."""
    if original.get('review_status')=='reviewed' or 'editor_draft' in original:
        raise ValueError('Never replace reviewed labels or an unfinished draft')
    pdf_source(original)
    if result.get('input_type') != 'pdf' or result.get('source_sha256') != original['source_sha256'] or result.get('source') != original['source']:
        raise ValueError('Draft does not match the original PDF')
    validate_invoice(result['extraction'])
    incoming = copy.deepcopy(original)
    incoming['_disk_revision'] = revision(original)
    incoming['annotation'] = result['extraction']
    incoming['auto_labeled'] = True
    incoming['verification'] = {'status':'pending'}
    incoming['review_status'] = 'needs_review'
    incoming['review'] = {'reviewed_at':None,'action':'gemini_draft','model':result['model'],
                          'input_type':'pdf','source_sha256':original['source_sha256'],
                          'prompt_sha256':result['prompt_sha256']}
    store.update(incoming)


def auto_labeled(record):
    review = record.get('review') or {}
    return (record.get('auto_labeled') is True or review.get('action')=='gemini_draft'
            or str(review.get('model','')).startswith('gemini-'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-file',type=Path,default=ROOT/'.env.labeling')
    parser.add_argument('--split',choices=('train','valid','test'),default='train')
    parser.add_argument('--groundtruth',type=Path,default=ROOT/'groundtruth')
    parser.add_argument('--output',type=Path,help='Default: outputs/labeling/generate/<split>')
    parser.add_argument('--model',help='Overrides GEMINI_MODEL')
    parser.add_argument('--document-id',help='Process one particular invoice')
    parser.add_argument('--limit',type=int,help='Maximum number of invoices, not images')
    parser.add_argument('--max-tokens',type=int,default=8192)
    parser.add_argument('--plan',action='store_true',help='List invoices without SDK, key or API calls')
    parser.add_argument('--apply',action='store_true',help='Save successful drafts into the HTML review dataset; never mark reviewed')
    parser.add_argument('--only-auto-labeled',action='store_true',help='Only automatically labeled invoices still awaiting human review')
    parser.add_argument('--only-unlabeled',action='store_true',help='Only invoices never automatically labeled; skip reviewed and unfinished drafts')
    parser.add_argument('--overwrite',action='store_true',help='Regenerate cached API results; never overrides reviewed invoices')
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit<1 or args.max_tokens<1:
        parser.error('limit/max-tokens must be positive')
    if args.only_unlabeled and args.only_auto_labeled:
        parser.error('Choose only one of --only-unlabeled and --only-auto-labeled')
    settings = read_settings(args.env_file)
    model = args.model or settings.get('GEMINI_MODEL') or DEFAULT_MODEL
    interval = float(settings.get('GEMINI_REQUEST_INTERVAL') or 5)
    if not 0 <= interval < float('inf'):
        parser.error('GEMINI_REQUEST_INTERVAL must be finite and nonnegative')
    prompt = (ROOT/'prompts/invoice.txt').read_text(encoding='utf-8')
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
    records = [r for r in read_jsonl(args.groundtruth/'annotations'/(args.split+'.jsonl'))
               if r.get('review_status')!='reviewed' and 'editor_draft' not in r
               and (not args.document_id or r['document_id']==args.document_id)
               and (not args.only_unlabeled or not auto_labeled(r))
               and (not args.only_auto_labeled or auto_labeled(r))]
    pdf_records = []
    for record in records:
        if Path(record['source']).suffix.lower() != '.pdf':
            print(record['document_id'], 'SKIP: not a PDF:', record['source'], flush=True)
            continue
        pdf_records.append(record)
    records = pdf_records
    if args.limit:
        records = records[:args.limit]
    print(f'{len(records)} pending invoices; model={model}',flush=True)
    if args.plan:
        for r in records:
            print(r['document_id'], 'PDF', r['source'], flush=True)
        return 0
    cache_root = ROOT/'gemini/outputs' if (ROOT/'gemini/outputs').is_dir() else ROOT/'outputs/labeling/generate'
    output = args.output or cache_root/args.split
    store = Store(args.groundtruth) if args.apply else None
    client = None
    errors = 0
    last_request_finished = None
    try:
        for record in records:
            path = output/(record['document_id']+'.json')
            result = None
            try:
                pdf_source(record)
            except (ValueError, OSError) as error:
                errors += 1
                print(record['document_id'], 'ERROR:', error, flush=True)
                continue
            try:
                if path.exists() and not args.overwrite:
                    previous = loads_exact(path.read_text(encoding='utf-8'))
                    if (previous.get('model')!=model or previous.get('prompt_sha256')!=prompt_hash
                            or previous.get('input_type')!='pdf' or previous.get('source')!=record['source']
                            or previous.get('source_sha256')!=record['source_sha256']
                            or previous.get('images_sha256')!=images_digest(record['images'])):
                        if not args.only_unlabeled:
                            raise ValueError('Cached result uses another model/prompt/PDF; use a new --output or --overwrite')
                        # This invoice passed the never-auto-labeled filter. Its stale
                        # cache is not an applied label; regenerate using current settings.
                        previous = {}
                    if previous.get('status')=='ok':
                        validate_invoice(previous['extraction'])
                        result = previous
                if result is None:
                    if client is None:
                        key = settings.get('GEMINI_API_KEY')
                        if not key:
                            raise ValueError('Set GEMINI_API_KEY in .env.labeling or environment')
                        try:
                            from google import genai
                            from google.genai import types
                        except ImportError:
                            raise ValueError('Install SDK: python -m pip install -U google-genai') from None
                        client = genai.Client(api_key=key)
                    if last_request_finished is not None:
                        time.sleep(max(0, interval-(time.monotonic()-last_request_finished)))
                    result = extract_one(client,types,record,model,prompt,args.max_tokens)
                    if result.get('request_sent'):
                        last_request_finished = time.monotonic()
                    write_json(path,result)
                if result['status']!='ok':
                    raise ValueError(result.get('error','Extraction failed'))
                if store:
                    apply_draft(store,record,result)
                print(record['document_id'],'draft saved'+(' and applied to review' if store else ''),flush=True)
            except (ValueError,OSError) as error:
                errors += 1
                print(record['document_id'],'ERROR:',error,flush=True)
                if client is None or (result or {}).get('retryable'):
                    break
    finally:
        if client is not None:
            client.close()
    return 1 if errors else 0

if __name__=='__main__':
    sys.exit(main())
