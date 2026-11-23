"""Prepare demo image invoices and preserve existing invoice labels. Never used by inference."""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
from invoice_ocr.config import ROOT
from invoice_ocr.storage.files import EXTENSIONS, file_sha256, read_jsonl, write_jsonl
from invoice_ocr.contracts.invoice import TEXT_KEYS, VEHICLE_KEYS, ITEM_KEYS, validate_invoice


def blank():
    return {**dict.fromkeys(TEXT_KEYS), 'vehicle_information': dict.fromkeys(VEHICLE_KEYS),
            'items': [], 'totals': {'subtotal': None, 'taxes': [], 'total_payment': None}}


def render(source, destination):
    """200 DPI RGB JPEG, quality=95. Never crop or OCR the source."""
    from PIL import Image, ImageOps
    destination.mkdir(parents=True, exist_ok=True)
    paths = []
    if source.suffix.lower() == '.pdf':
        import pymupdf
        with pymupdf.open(source) as pdf:
            if pdf.needs_pass:
                raise ValueError('Encrypted PDF: ' + str(source))
            for index, page in enumerate(pdf, 1):
                target = destination / f'page_{index:03}.jpg'
                if not target.exists():
                    pix = page.get_pixmap(dpi=200, colorspace=pymupdf.csRGB, alpha=False, annots=True)
                    image = Image.frombytes('RGB', (pix.width, pix.height), pix.samples)
                    image.save(target, quality=95, subsampling=0, dpi=(200,200))
                    image.close()
                paths.append(target)
    else:
        with Image.open(source) as original:
            for index in range(getattr(original, 'n_frames', 1)):
                original.seek(index)
                target = destination / f'page_{index+1:03}.jpg'
                if not target.exists():
                    image = ImageOps.exif_transpose(original).convert('RGB')
                    image.save(target, quality=95, subsampling=0, dpi=(200,200))
                    image.close()
                paths.append(target)
    if not paths:
        raise ValueError('Empty document: ' + str(source))
    return paths


def build(output):
    output = output.resolve()
    annotations = output / 'annotations'
    annotations.mkdir(parents=True, exist_ok=True)
    existing = {}
    for split in ('train','valid','test'):
        path = annotations/(split+'.jsonl')
        if path.exists():
            for record in read_jsonl(path):
                if record.get('label_format') != 'invoice-v3':
                    raise ValueError('Expected current invoice-v3 labels; refusing to overwrite: '+str(path))
                existing[record['source']] = record
    generated = {}
    for split in ('train','valid','test'):
        records = []
        source_root = ROOT/'data_split'/split
        sources = sorted(p for p in source_root.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS | {'.pdf'})
        for index, source in enumerate(sources, 1):
            relative = source.relative_to(ROOT).as_posix()
            sha = file_sha256(source)
            previous = existing.get(relative)
            document_id = (previous or {}).get('document_id') or hashlib.sha256((split+'\n'+relative+'\n'+sha).encode()).hexdigest()[:20]
            if previous and previous['source_sha256'] != sha:
                raise ValueError('Original source changed; refusing to overwrite labels: ' + relative)
            paths = render(source, ROOT/'data'/split/document_id)
            images = [{'file': p.relative_to(ROOT).as_posix(), 'sha256': file_sha256(p)} for p in paths]
            if previous:
                if previous['images'] != images:
                    raise ValueError('Image content changed; refusing to relink labels: ' + relative)
                record = previous
            else:
                record = {'label_format': 'invoice-v3', 'document_id': document_id, 'split': split,
                          'source': relative, 'source_sha256': sha, 'vendor_folder': source.parent.name,
                          'images': images, 'review_status': 'needs_review', 'annotation': blank(),
                          'verification': {'status':'unlabeled'},
                          'review': {'reviewed_at': None}}
            records.append(record)
            if index % 20 == 0 or index == len(sources):
                print(f'{split}: {index}/{len(sources)} documents converted', flush=True)
        generated[split] = records
    for split, records in generated.items():
        write_jsonl(annotations/(split+'.jsonl'), records)
        print(split, len(records), 'invoices,', sum(len(r['images']) for r in records), 'images,',
              sum(r['review_status']=='reviewed' for r in records), 'reviewed', flush=True)
    from invoice_ocr.labeling.review import REVIEW_HTML
    (output/"review.html").write_text(REVIEW_HTML, encoding="utf-8")
    print('Original PDFs preserved. Start: invoice-ocr label review', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'groundtruth')
    args = parser.parse_args(argv)
    build(args.output)

if __name__ == '__main__':
    main()
