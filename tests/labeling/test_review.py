import json
from pathlib import Path
import tempfile
import unittest
from invoice_ocr.config import ROOT
from invoice_ocr.contracts import invoice as schema
from tests.fixtures import typed_invoice


class InvoiceReviewTests(unittest.TestCase):
    def setUp(self):
        from invoice_ocr.labeling.review import Store
        from invoice_ocr.storage.files import write_jsonl
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory=Path(self.temp.name)
        self.record={'label_format':'invoice-v3','document_id':'id1','split':'train','source':'demo.pdf',
            'source_sha256':'f'*64,'images':[{'file':'data/train/id1/page_001.jpg','sha256':'a'*64},
            {'file':'data/train/id1/page_002.jpg','sha256':'b'*64}],
            'annotation':schema.parse_invoice(json.dumps(typed_invoice())),
            'review_status':'needs_review','field_review':{}}
        write_jsonl(self.directory/'annotations/train.jsonl',[self.record])
        self.store=Store(self.directory)

    def editable(self):
        from invoice_ocr.labeling.review import revision
        record=self.store.read('train')[0]
        record['_disk_revision']=revision(record)
        return record

    def test_one_update_reviews_the_whole_multi_image_invoice(self):
        record=self.editable();record['review_status']='reviewed'
        self.store.update(record)
        saved=self.store.read('train')[0]
        self.assertEqual(saved['review_status'],'reviewed')
        self.assertEqual(len(saved['images']),2)
        self.assertNotIn('pages',saved)
        from invoice_ocr.storage.files import read_jsonl
        subset=read_jsonl(self.directory/'reviewed/train.jsonl')
        self.assertEqual(len(subset),1)
        self.assertEqual(subset[0]['annotation'],saved['annotation'])
        self.assertEqual(set(subset[0]),{'document_id','images','annotation'})
        self.assertTrue(all(isinstance(image,str) for image in subset[0]['images']))

    def test_reviewed_subset_updates_when_editing_or_unreviewing(self):
        from invoice_ocr.storage.files import read_jsonl
        record=self.editable();record['review_status']='reviewed'
        self.store.update(record)
        record=self.editable();record['annotation']['invoice_id']='updated'
        self.store.update(record)
        self.assertEqual(read_jsonl(self.directory/'reviewed/train.jsonl')[0]['annotation']['invoice_id'],'updated')
        record=self.editable();record['review_status']='needs_review'
        self.store.update(record)
        self.assertEqual(read_jsonl(self.directory/'reviewed/train.jsonl'),[])

    def test_changed_image_identity_is_rejected(self):
        record=self.editable();record['images'].reverse()
        with self.assertRaisesRegex(ValueError,'identity'):
            self.store.update(record)

    def test_invalid_annotation_does_not_modify_disk(self):
        before=(self.directory/'annotations/train.jsonl').read_bytes()
        record=self.editable();record['annotation']['items'][0]['line_total']='529.00'
        with self.assertRaises(ValueError):self.store.update(record)
        self.assertEqual((self.directory/'annotations/train.jsonl').read_bytes(),before)

    def test_concurrent_update_is_rejected(self):
        record=self.editable();record['annotation']['invoice_id']='new'
        self.store.update(record)
        with self.assertRaisesRegex(ValueError,'phiên khác'):
            self.store.update(record)
