"""Image document discovery and exact JSON access. No PDF handling at inference."""
from __future__ import annotations
from ..config import ROOT
import hashlib
import json
import tempfile
import time
from decimal import Decimal
from pathlib import Path

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.tif', '.tiff'}


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def loads_exact(text):
    def reject(value):
        raise ValueError('Invalid JSON number: ' + value)
    return json.loads(text, parse_float=Decimal, parse_constant=reject)


def dumps_exact(value):
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError('Non-finite JSON number')
        return str(value)
    if isinstance(value, dict):
        return '{' + ','.join(json.dumps(k, ensure_ascii=False) + ':' + dumps_exact(v) for k, v in value.items()) + '}'
    if isinstance(value, (list, tuple)):
        return '[' + ','.join(dumps_exact(v) for v in value) + ']'
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def _write_atomic(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Close each unique temporary file before replacing on Windows.
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                     prefix=path.name+'.', suffix='.tmp', delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(text)
        except BaseException:
            stream.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        for attempt in range(11):
            try:
                temporary.replace(path)
                return
            except PermissionError:
                if attempt == 10:
                    raise
                # Editors/indexers can briefly prevent replacement.
                time.sleep(0.2)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, value):
    _write_atomic(path, dumps_exact(value))


def write_jsonl(path, records):
    _write_atomic(path, ''.join(dumps_exact(r) + '\n' for r in records))


def read_jsonl(path):
    return [loads_exact(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def image_entries(images):
    """Resolve clean image-path lists; keep internal hashed entries compatible."""
    root = ROOT
    entries = []
    for image in images:
        if isinstance(image, str):
            path = (root/image).resolve()
            if not path.is_relative_to((root/'data').resolve()) or not path.is_file():
                raise ValueError('Invalid image path: '+image)
            entries.append({'file': image, 'sha256': file_sha256(path)})
        elif isinstance(image, dict) and 'file' in image and 'sha256' in image:
            entries.append(image)
        else:
            raise ValueError('Each image must be a path or an internal hashed image entry')
    if not entries:
        raise ValueError('Invoice must contain images')
    return entries


def documents(source):
    """One immediate image folder is one invoice. A single image is also accepted."""
    source = source.resolve()
    if not source.exists():
        raise ValueError(f'Input does not exist: {source}')
    if source.is_file() and source.suffix.lower() == '.jsonl':
        root = ROOT
        invoices = []
        for record in read_jsonl(source):
            if record.get('label_format', 'invoice-v3') != 'invoice-v3' or not record.get('images'):
                raise ValueError('Expected invoice-v3 image manifest')
            entries = image_entries(record['images'])
            paths = [(root / image['file']).resolve() for image in entries]
            document = paths[0].parent
            if document.name != record['document_id'] or any(p.parent != document or not p.is_relative_to((root/'data').resolve()) for p in paths):
                raise ValueError('Manifest invoice images must share their document_id folder under data/')
            if paths != image_paths(document):
                raise ValueError('Manifest must include every invoice image in order')
            for path, image in zip(paths, entries):
                if file_sha256(path) != image['sha256']:
                    raise ValueError('Image hash changed: ' + str(path))
            invoices.append(document)
        if not invoices:
            raise ValueError('No invoices in manifest')
        if len({p.name for p in invoices}) != len(invoices):
            raise ValueError('Duplicate document_id in manifest')
        return source.parent, invoices
    if source.is_file():
        if source.suffix.lower() not in EXTENSIONS:
            raise ValueError('Inference accepts image files only, never PDF')
        return source.parent, [source]
    if list(source.rglob('*.pdf')) or list(source.rglob('*.PDF')):
        raise ValueError('PDF is not an inference input. Convert demo files with prepare_groundtruth.py first.')
    direct = [p for p in source.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS]
    folders = sorted({p.parent for p in source.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS})
    if direct and len(folders) > 1:
        raise ValueError('Do not mix invoice images and invoice subfolders in one input')
    if direct:
        return source.parent, [source]
    if not folders:
        raise ValueError('No invoice image folders found')
    if len({p.name for p in folders}) != len(folders):
        raise ValueError('Invoice folder names must be unique document_id values')
    return source, folders


def image_paths(document):
    paths = [document] if document.is_file() else sorted(p for p in document.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if not paths:
        raise ValueError('Invoice contains no images')
    return paths


def images_digest(images):
    """Identity depends on ordered image content, not filesystem paths."""
    return hashlib.sha256('\n'.join(item['sha256'] for item in images).encode('ascii')).hexdigest()


def load_images(document, max_edge):
    from PIL import Image, ImageOps
    frames = []
    try:
        for path in image_paths(document):
            with Image.open(path) as source:
                if getattr(source, 'n_frames', 1) != 1:
                    raise ValueError('Use one image file per page; split multi-frame images first')
                frame = ImageOps.exif_transpose(source).convert('RGB')
                frame.thumbnail((max_edge, max_edge))
                frames.append(frame)
        return frames
    except Exception:
        for frame in frames:
            frame.close()
        raise


def json_records(source, normalized=False):
    source = source.resolve()
    if not source.exists():
        raise ValueError(f'Input does not exist: {source}')
    base = source.parent if source.is_file() else source
    paths = [source] if source.is_file() else sorted(source.rglob('*.json'))
    found = []
    for path in paths:
        data = loads_exact(path.read_text(encoding='utf-8-sig'))
        if isinstance(data, dict) and (data.get('label_format') == 'invoice-v3' or ('extraction' in data and not normalized) or isinstance(data.get('pages'), list) and ('normalization' in data) == normalized):
            found.append((path, data))
    if not found:
        raise ValueError('No matching prediction JSON records found')
    return base, found
