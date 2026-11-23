"""Export reviewed invoice labels; optionally package their images and code for Kaggle."""
from __future__ import annotations
import argparse
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from invoice_ocr.config import ROOT
from invoice_ocr.storage.files import read_jsonl, write_jsonl, dumps_exact, file_sha256, image_entries
from invoice_ocr.contracts.invoice import validate_invoice


def reviewed_records(records):
    """Build the reviewed subset without copying or rereading image files."""
    selected = []
    for record in records:
        if record.get('review_status') != 'reviewed' or 'editor_draft' in record:
            continue
        validate_invoice(record['annotation'])
        item = {'document_id': record['document_id'],
                'images': [image['file'] for image in record['images']],
                'annotation': record['annotation']}
        selected.append(item)
    return selected


def export(directory, splits=('train','valid','test')):
    directory = directory.resolve()
    for split in splits:
        source_records = read_jsonl(directory/'annotations'/(split+'.jsonl'))
        records = reviewed_records(source_records)
        for record in source_records:
            if record.get('review_status') != 'reviewed' or 'editor_draft' in record:
                continue
            for image in record['images']:
                path = (ROOT/image['file']).resolve()
                if not path.is_relative_to((ROOT/'data'/split).resolve()) or file_sha256(path) != image['sha256']:
                    raise ValueError('Image missing or changed: '+image['file'])
        write_jsonl(directory/'reviewed'/(split+'.jsonl'), records)
        print(f'{split}: {len(records)} reviewed invoices, {sum(len(r["images"]) for r in records)} images', flush=True)


def package(directory, split):
    records = read_jsonl(directory/'reviewed'/(split+'.jsonl'))
    if not records:
        print(f'{split}: no reviewed invoice; ZIP not created', flush=True)
        return None
    archive = ROOT/('invoice-reviewed-'+split+'.zip')
    temporary = archive.with_suffix('.zip.tmp')
    # Whitelist source/assets; never include .env, local caches or unrelated labels.
    patterns = {'src': '*.py', 'app': '*.py', 'scripts': '*.py', 'database': '*.sql',
                'requirements': '*.txt', 'configs': '*.env.example', 'prompts': '*.txt'}
    files = [p for directory, pattern in patterns.items() for p in (ROOT/directory).rglob(pattern)]
    files.append(ROOT/'pyproject.toml')
    with ZipFile(temporary, 'w', ZIP_DEFLATED) as z:
        for path in files:
            if path.is_file():
                z.write(path, path.relative_to(ROOT).as_posix())
        seen = set()
        for record in records:
            for image in image_entries(record['images']):
                path = (ROOT/image['file']).resolve()
                if not path.is_relative_to((ROOT/'data'/split).resolve()) or file_sha256(path) != image['sha256']:
                    raise ValueError('Image identity changed: '+image['file'])
                if image['file'] not in seen:
                    z.write(path, image['file'])
                    seen.add(image['file'])
        z.writestr('groundtruth/reviewed/'+split+'.jsonl', ''.join(dumps_exact(r)+'\n' for r in records))
    temporary.replace(archive)
    print('ZIP:', archive, flush=True)
    return archive


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--groundtruth', type=Path, default=ROOT/'groundtruth')
    parser.add_argument('--split', choices=('train','valid','test','all'), default='all')
    parser.add_argument('--zip', action='store_true')
    args = parser.parse_args(argv)
    splits = ('train','valid','test') if args.split=='all' else (args.split,)
    export(args.groundtruth, splits)
    if args.zip:
        for split in splits:
            package(args.groundtruth, split)

if __name__ == '__main__':
    main()
