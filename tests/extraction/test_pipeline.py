import contextlib
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
from decimal import Decimal
from invoice_ocr.config import ROOT
from invoice_ocr.extraction import pipeline as extraction
from tests.fixtures import invoice, typed_invoice


class OutputHandlingTests(unittest.TestCase):
    def setUp(self):
        (ROOT/'tmp').mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=ROOT/'tmp')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root/'input'
        self.invoice_dir = self.source/'invoice_01'
        self.invoice_dir.mkdir(parents=True)
        (self.invoice_dir/'page_001.jpg').write_bytes(b'first-image')
        (self.invoice_dir/'page_002.jpg').write_bytes(b'second-image')
        self.output = self.root/'output'
        self.args = types.SimpleNamespace(input=self.source, output=self.output, overwrite=False, limit=None,
            prompt=ROOT/'prompts/invoice.txt', model='mock-model', revision='main', max_edge=1600, max_new_tokens=4096)

    def test_one_job_per_image_invoice(self):
        self.assertEqual(extraction.jobs(self.args), [(self.invoice_dir,self.output/'invoice_01.json')])

    def test_pdf_input_is_rejected(self):
        (self.source/'invoice.pdf').write_bytes(b'%PDF')
        with self.assertRaisesRegex(ValueError,'PDF'):
            extraction.jobs(self.args)

    def test_existing_output_is_rejected(self):
        self.output.mkdir()
        (self.output/'invoice_01.json').write_text('keep')
        with self.assertRaisesRegex(ValueError,'Output exists'):
            extraction.jobs(self.args)

    def test_nested_output_is_rejected(self):
        self.args.output=self.source/'output'
        with self.assertRaisesRegex(ValueError,'outside'):
            extraction.jobs(self.args)

    def run_fake_extraction(self,text,tokens=10):
        class Inputs(dict):
            def to(self, device):return self
        class Output:
            def __getitem__(self,key):return list(range(tokens))
        self.messages=[]
        def template(messages,**kwargs):
            self.messages.append(messages)
            return Inputs(input_ids=types.SimpleNamespace(shape=(1,5)))
        processor=types.SimpleNamespace(apply_chat_template=template,decode=lambda *args,**kwargs:text)
        model=types.SimpleNamespace(device='cuda:0',generate=lambda **kwargs:Output())
        torch=types.SimpleNamespace(cuda=types.SimpleNamespace(synchronize=lambda:None,empty_cache=lambda:None),inference_mode=contextlib.nullcontext)
        with patch.dict('sys.modules',{'torch':torch}):
            code=extraction.extract(self.args,model_loader=lambda args:(model,processor),image_loader=lambda *args:[types.SimpleNamespace(close=lambda:None),types.SimpleNamespace(close=lambda:None)])
        return code,json.loads((self.output/'invoice_01.json').read_text(encoding='utf-8'),parse_float=Decimal)

    def test_all_images_are_sent_once_and_one_json_saved(self):
        code,result=self.run_fake_extraction(json.dumps(typed_invoice()))
        self.assertEqual(code,0)
        self.assertEqual(len(self.messages),1)
        self.assertEqual(sum(c['type']=='image' for c in self.messages[0][0]['content']),2)
        self.assertEqual(len(result['images']),2)
        self.assertNotIn('pages',result)
        self.assertEqual(result['extraction']['totals']['total_payment'],Decimal('597.77'))

    def test_invalid_json_keeps_raw_text(self):
        code,result=self.run_fake_extraction('invalid JSON')
        self.assertEqual(code,1)
        self.assertEqual(result['raw_text'],'invalid JSON')
        self.assertNotIn('extraction',result)

    def test_token_limit_is_not_success(self):
        code,result=self.run_fake_extraction(json.dumps(typed_invoice()),tokens=4096)
        self.assertEqual(code,1)
        self.assertIn('token limit',result['error'])

    def test_numeric_strings_are_rejected_without_coercion(self):
        text=json.dumps(invoice())
        code,result=self.run_fake_extraction(text)
        self.assertEqual(code,1)
        self.assertEqual(result['raw_text'],text)

    def test_long_decimal_is_preserved(self):
        text=json.dumps(typed_invoice()).replace('597.77','12345678901234.1234')
        code,result=self.run_fake_extraction(text)
        self.assertEqual(code,0)
        self.assertEqual(result['extraction']['totals']['total_payment'],Decimal('12345678901234.1234'))

    def test_non_iso_date_is_rejected(self):
        data=typed_invoice();data['date']='03/04/2025'
        code,result=self.run_fake_extraction(json.dumps(data))
        self.assertEqual(code,1)
        self.assertIn('ISO date',result['error'])
