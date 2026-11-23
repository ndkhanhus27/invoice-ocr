"""Upload orchestration checks, without downloading a model or requiring CUDA."""
import sys
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[2]
from app.service import UploadService
from invoice_ocr.storage.files import write_json, loads_exact
from invoice_ocr.contracts.invoice import TEXT_KEYS, VEHICLE_KEYS


class UploadTests(unittest.TestCase):
    def test_upload_keeps_page_order_and_saves_exact_result(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = [root/'second.png', root/'first.png']
            for index, file in enumerate(files):
                Image.new('RGB', (2, 2), (index * 100, 0, 0)).save(file)
            calls = []
            def extract(args, model_loader):
                calls.append([p.read_bytes() for p in sorted(args.input.iterdir())])
                model_loader(args)
                data = {**dict.fromkeys(TEXT_KEYS), 'purchase_order': '00042',
                        'vehicle_information': dict.fromkeys(VEHICLE_KEYS), 'items': [],
                        'totals': {'subtotal': None, 'taxes': [], 'total_payment': 42}}
                write_json(args.output/(args.input.name+'.json'),
                           {'label_format': 'invoice-v3', 'status': 'ok', 'extraction': data})
            loads = []
            service = UploadService(SimpleNamespace(output=root/'output'),
                                    model_loader=lambda args: loads.append(1) or ('model', 'processor'),
                                    extractor=extract)
            for _ in range(2):
                status, text, download, report = service.run(list(map(str, files)))
                self.assertIn('00042', status)
                self.assertEqual(loads_exact(text)['totals']['total_payment'], 42)
                self.assertEqual(loads_exact(Path(download).read_text()), report)
            self.assertEqual(len(loads), 1)
            self.assertEqual(calls[0], [p.read_bytes() for p in files])

    def test_empty_upload_and_save_without_result_are_rejected(self):
        service = UploadService(SimpleNamespace())
        with self.assertRaises(ValueError):
            service.run([])
        with self.assertRaises(ValueError):
            service.save(None)

    def test_invalid_image_does_not_call_model(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/'fake.png'
            source.write_text('not an image')
            service = UploadService(SimpleNamespace(output=Path(directory)/'output'),
                                    model_loader=lambda args: self.fail('Model must not load'))
            from PIL import UnidentifiedImageError
            with self.assertRaises(UnidentifiedImageError):
                service.run([str(source)])
            self.assertFalse((Path(directory)/'output').exists())


if __name__ == '__main__':
    unittest.main()
