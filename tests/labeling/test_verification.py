"""Local verification tests; every provider request is mocked."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock
from types import SimpleNamespace
from decimal import Decimal
ROOT = Path(__file__).resolve().parents[2]
from invoice_ocr.storage.files import write_jsonl, read_jsonl, dumps_exact, images_digest, file_sha256
from invoice_ocr.contracts.invoice import TEXT_KEYS, VEHICLE_KEYS
from invoice_ocr.labeling.comparison import compare_annotations, has_content
from invoice_ocr.labeling.review import Store, revision
from invoice_ocr.labeling import verify as checker, gemini as labeler


def label():
    return {**dict.fromkeys(TEXT_KEYS),'company_name':'Seller LTD','invoice_id':'0001',
            'vehicle_information':dict.fromkeys(VEHICLE_KEYS),
            'items':[{'item_code':'A','description':'Part A','quantity':2,
                      'unit_price':Decimal('5.00'),'line_total':Decimal('10.00')}],
            'totals':{'subtotal':10,'taxes':[],'total_payment':10}}


class JudgmentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.directory = self.root/'groundtruth'
        source = self.root/'data_split/train/demo.pdf'
        source.parent.mkdir(parents=True)
        source.write_bytes(b'%PDF-1.7 fake original document')
        root_patch = patch.object(labeler, 'ROOT', self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)
        images = []
        for i in (1,2):
            path = self.root/f'data/train/doc/page_{i:03}.jpg'
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(b'fake-image-'+str(i).encode())
            images.append({'file':path.relative_to(self.root).as_posix(),'sha256':file_sha256(path)})
        self.original = {'label_format':'invoice-v3','document_id':'doc','split':'train',
            'source':'data_split/train/demo.pdf','source_sha256':file_sha256(source),'images':images,'annotation':label(),
            'auto_labeled':True,'review_status':'needs_review','verification':{'status':'pending'},
            'review':{'action':'pending_verification','model':'gemini-3.1-flash-lite','reviewed_at':None,
                      'input_type':'pdf','source_sha256':file_sha256(source),'prompt_sha256':'p'*64}}
        write_jsonl(self.directory/'annotations/train.jsonl',[self.original])
        self.store = Store(self.directory)

    def result(self, annotation=None, status='ok'):
        return {'model':'gemini-3.5-flash-lite','prompt_sha256':'p'*64,'status':status,
                'input_type':'pdf','source':self.original['source'],'source_sha256':self.original['source_sha256'],
                'images':self.original['images'],'images_sha256':images_digest(self.original['images']),
                'extraction':annotation if annotation is not None else label(), 'error':'Provider failed'}

    def test_normalized_text_and_exact_decimal_agree(self):
        other=label();other['company_name']=' seller ltd ';other['items'][0]['line_total']=10
        self.assertEqual(compare_annotations(label(),other),[])

    def test_wrong_cent_and_missing_word_are_flagged(self):
        other=label();other['items'][0]['line_total']=Decimal('10.01');other['company_name']='Seller'
        self.assertEqual({d['field'] for d in compare_annotations(label(),other)},
                         {'items[0].line_total','company_name'})

    def test_agreement_auto_reviews_and_exports_without_metadata(self):
        self.assertEqual(checker.apply_judgment(self.store,self.original,self.result()),'agreed')
        saved=self.store.read('train')[0]
        self.assertEqual(saved['review_status'],'reviewed')
        self.assertEqual(saved['review']['action'],'automatic_agreement')
        exported=read_jsonl(self.directory/'reviewed/train.jsonl')
        self.assertEqual(set(exported[0]),{'document_id','images','annotation'})
        self.assertEqual(saved['annotation'],self.original['annotation'])

    def test_disagreement_preserves_label_and_waits_for_human(self):
        candidate=label();candidate['invoice_id']='1'
        checker.apply_judgment(self.store,self.original,self.result(candidate))
        saved=self.store.read('train')[0]
        self.assertEqual(saved['review_status'],'needs_review')
        self.assertEqual(saved['annotation']['invoice_id'],'0001')
        self.assertEqual(saved['verification']['candidate']['invoice_id'],'1')
        self.assertEqual(saved['verification']['differences'][0]['field'],'invoice_id')
        self.assertEqual(read_jsonl(self.directory/'reviewed/train.jsonl'),[])

    def test_provider_error_never_approves(self):
        checker.apply_judgment(self.store,self.original,self.result(status='error'))
        saved=self.store.read('train')[0]
        self.assertEqual(saved['verification']['status'],'error')
        self.assertEqual(saved['review_status'],'needs_review')

    def test_same_model_is_rejected(self):
        result=self.result();result['model']='gemini-3.1-flash-lite'
        with self.assertRaisesRegex(ValueError,'different model'):
            checker.apply_judgment(self.store,self.original,result)

    def test_human_save_during_request_is_not_overwritten(self):
        incoming=copy.deepcopy(self.original);incoming['_disk_revision']=revision(self.original)
        incoming['annotation']['company_name']='Human correction'
        self.store.update(incoming)
        with self.assertRaisesRegex(ValueError,'phi'):
            checker.apply_judgment(self.store,self.original,self.result())
        self.assertEqual(self.store.read('train')[0]['annotation']['company_name'],'Human correction')

    def test_human_can_review_a_disagreement(self):
        candidate=label();candidate['invoice_id']='1'
        checker.apply_judgment(self.store,self.original,self.result(candidate))
        saved=self.store.read('train')[0];saved['_disk_revision']=revision(saved)
        saved['review_status']='reviewed';saved['review']['action']='user_update_invoice'
        self.store.update(saved)
        self.assertEqual(len(read_jsonl(self.directory/'reviewed/train.jsonl')),1)
        self.assertEqual(self.store.read('train')[0]['verification']['status'],'disagreed')

    def test_reset_preserves_values_and_clears_reviewed(self):
        checker.apply_judgment(self.store,self.original,self.result())
        self.assertEqual(self.store.reset(('train',)),{'train':1})
        saved=self.store.read('train')[0]
        self.assertEqual(saved['annotation'],self.original['annotation'])
        self.assertEqual(saved['verification'],{'status':'pending'})
        self.assertEqual(saved['review_status'],'needs_review')
        self.assertEqual(read_jsonl(self.directory/'reviewed/train.jsonl'),[])

    def test_unreview_removes_auto_approved_record_from_export(self):
        checker.apply_judgment(self.store,self.original,self.result())
        saved=self.store.read('train')[0];saved['_disk_revision']=revision(saved)
        saved['review_status']='needs_review';saved['review']['action']='user_unreview_invoice'
        self.store.update(saved)
        self.assertEqual(read_jsonl(self.directory/'reviewed/train.jsonl'),[])



    def test_two_empty_labels_are_not_automatically_accepted(self):
        empty={**dict.fromkeys(TEXT_KEYS),'vehicle_information':dict.fromkeys(VEHICLE_KEYS),
               'items':[],'totals':{'subtotal':None,'taxes':[],'total_payment':None}}
        self.assertFalse(has_content(empty))
        self.assertEqual(compare_annotations(empty,empty)[0]['field'],'__evidence__')

    def test_request_contains_original_pdf_and_no_prior_answer_or_images(self):
        client = MagicMock()
        client.models.generate_content.return_value = SimpleNamespace(
            text=dumps_exact(label()), candidates=[SimpleNamespace(finish_reason='STOP')])
        types = MagicMock()
        types.Part.from_bytes.side_effect = lambda **kw: kw
        self.original['annotation']['invoice_id'] = 'PRIVATE-PRIOR-ANSWER'
        result = labeler.extract_one(client, types, self.original, 'gemini-3.5-flash-lite', 'shared rules', 8192)
        self.assertEqual(result['status'], 'ok')
        payload = client.models.generate_content.call_args.kwargs
        self.assertEqual(payload['contents'], ['shared rules', {
            'data':(self.root/self.original['source']).read_bytes(), 'mime_type':'application/pdf'}])
        self.assertNotIn('PRIVATE-PRIOR-ANSWER', str(payload))
        self.assertNotIn('fake-image', str(payload))

    def test_changed_pdf_is_rejected_before_request(self):
        (self.root/self.original['source']).write_bytes(b'changed PDF')
        client = MagicMock()
        with self.assertRaisesRegex(ValueError, 'PDF missing or changed'):
            labeler.extract_one(client, MagicMock(), self.original, 'gemini-3.5-flash-lite', 'rules', 8192)
        client.models.generate_content.assert_not_called()

    def test_old_image_result_cannot_auto_approve(self):
        result = self.result()
        result.pop('input_type')
        with self.assertRaisesRegex(ValueError, 'original PDF'):
            checker.apply_judgment(self.store, self.original, result)

    def test_extra_item_is_a_disagreement(self):
        other=label();other['items'].append(copy.deepcopy(other['items'][0]))
        differences=compare_annotations(label(),other)
        self.assertEqual(len(differences),1)
        self.assertEqual(differences[0]['label_value'],'<missing>')










if __name__=='__main__':
    unittest.main()
