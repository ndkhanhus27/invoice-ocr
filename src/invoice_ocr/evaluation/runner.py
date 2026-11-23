"""Evaluate reviewed invoices with field-aware exact match and text CER."""
from __future__ import annotations
import argparse
from pathlib import Path
from invoice_ocr.config import ROOT
from invoice_ocr.storage.files import loads_exact as loads, images_digest, image_entries
from invoice_ocr.contracts.invoice import validate_invoice
from invoice_ocr.evaluation.scoring import MISSING, Scorer, same

def evaluate(groundtruth, predictions, output):
    records = [loads(line) for line in groundtruth.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    if not records:
        raise ValueError('Ground truth is empty: review an entire invoice first')
    if not predictions.exists():
        raise ValueError('Predictions path does not exist')
    candidates = [predictions] if predictions.is_file() else sorted(predictions.rglob('*.json'))
    by_id = {}
    for path in candidates:
        prediction = loads(path.read_text(encoding='utf-8-sig'))
        if not isinstance(prediction, dict) or prediction.get('label_format') != 'invoice-v3':
            raise ValueError('Expected image-invoice prediction, not legacy per-page output: '+str(path))
        document_id = prediction.get('document_id')
        if not document_id:
            raise ValueError('Missing document_id: '+str(path))
        if document_id in by_id:
            raise ValueError('Multiple predictions for one invoice; choose one run: '+document_id)
        by_id[document_id] = prediction
    scorer = Scorer()
    for truth in records:
        if truth.get('review_status', 'reviewed') != 'reviewed' or 'editor_draft' in truth:
            raise ValueError('Evaluation requires reviewed ground truth, not automatic drafts')
        if truth.get('label_format', 'invoice-v3') != 'invoice-v3':
            raise ValueError('Expected invoice-v3 ground truth')
        validate_invoice(truth['annotation'])
        scorer.context = {'document_id': truth['document_id'], 'source': truth.get('source', truth['document_id'])}
        prediction = by_id.get(truth['document_id'], {})
        annotation = prediction.get('extraction', {})
        try:
            if prediction.get('status') != 'ok':
                raise ValueError(prediction.get('error', 'Missing or failed invoice prediction'))
            expected_hash = images_digest(image_entries(truth['images']))
            if images_digest(prediction.get('images', [])) != expected_hash or prediction.get('images_sha256') != expected_hash:
                raise ValueError('Invoice image content/order does not match ground truth')
            validate_invoice(annotation)
            valid = True
        except (ValueError, KeyError, TypeError) as error:
            valid = False
            annotation = {}
            scorer.error('__invoice__', 'invalid_prediction', truth['annotation'], MISSING, str(error))
        scorer.counts['schema_valid']['correct' if valid else 'wrong'] += 1
        bad = scorer.invoice(truth, annotation)
        scorer.counts['invoice_all_fields']['correct' if valid and not bad else 'wrong'] += 1
    scorer.write(output)
    print(f'Evaluated {len(records)} invoices')
    print('Metrics:', output/'metrics.csv')
    print('Errors:', output/'errors.csv')
    return scorer


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--groundtruth',type=Path,default=ROOT/'groundtruth/reviewed/test.jsonl')
    parser.add_argument('--predictions',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/evaluation-test')
    args=parser.parse_args(argv)
    evaluate(args.groundtruth,args.predictions,args.output)

if __name__=='__main__':main()
