"""Upload validation, extraction and database orchestration; no UI dependency."""
from __future__ import annotations

from copy import copy
from pathlib import Path
import shutil
import threading
from uuid import uuid4

from invoice_ocr.extraction.pipeline import extract
from invoice_ocr.storage.files import EXTENSIONS, dumps_exact, loads_exact
from invoice_ocr.extraction.backends.transformers import load_model
from invoice_ocr.storage.postgres import postgres_connection, store_report


class UploadService:
    def __init__(self, args, *, model_loader=load_model, extractor=extract):
        self.args = args
        self.model_loader = model_loader
        self.extractor = extractor
        self.model = None
        self.lock = threading.Lock()

    def cached_model(self, args):
        if self.model is None:
            self.model = self.model_loader(args)
        return self.model

    def run(self, files):
        if not files:
            raise ValueError('Chọn ít nhất một ảnh hóa đơn.')
        # Selection order is page order. Never trust uploaded names as storage paths.
        sources = [Path(path) for path in files]
        from PIL import Image
        for source in sources:
            if source.suffix.lower() not in EXTENSIONS:
                raise ValueError('Chỉ nhận JPG, PNG, WEBP, TIFF; mỗi file là một trang.')
            with Image.open(source) as image:
                if getattr(image, 'n_frames', 1) != 1:
                    raise ValueError('Tách ảnh nhiều frame thành từng trang trước khi upload.')
                image.verify()
        document_id = uuid4().hex
        work = self.args.output / 'uploads' / document_id
        document = work / 'images' / document_id
        document.mkdir(parents=True)
        for index, source in enumerate(sources, 1):
            shutil.copyfile(source, document / f'page_{index:04d}{source.suffix.lower()}')
        args = copy(self.args)
        args.input, args.output = document, work / 'raw'
        args.limit, args.overwrite = None, False
        # One model resident in VRAM; serialize all callers, including API clients.
        with self.lock:
            self.extractor(args, model_loader=self.cached_model)
        result_path = args.output / f'{document_id}.json'
        report = loads_exact(result_path.read_text(encoding='utf-8'))
        if report.get('status') != 'ok':
            return ('Trích xuất lỗi: ' + report.get('error', 'Không có kết quả'),
                    dumps_exact(report), str(result_path), None)
        data = report['extraction']
        status = (f"PO: {data['purchase_order'] or '—'} | "
                  f"Tổng thanh toán: {data['totals']['total_payment']} {data['currency'] or ''} | "
                  f"Số dòng: {len(data['items'])}")
        return status, dumps_exact(data), str(result_path), report

    def save(self, report):
        if not report or report.get('status') != 'ok':
            raise ValueError('Trích xuất thành công trước khi lưu PostgreSQL.')
        with postgres_connection(self.args) as connection:
            outcome = store_report(connection, report)
        return 'Đã lưu PostgreSQL.' if outcome == 'loaded' else 'Kết quả đã có trong PostgreSQL.'

