"""CPU tests for exact-match scoring; no model calls."""
import copy
import csv
from decimal import Decimal
from pathlib import Path
import sys
import tempfile
import unittest
from invoice_ocr.evaluation.runner import Scorer, evaluate, same
from invoice_ocr.contracts.invoice import TEXT_KEYS, VEHICLE_KEYS, ITEM_KEYS
from invoice_ocr.labeling.review import dumps
from invoice_ocr.storage.files import images_digest
from invoice_ocr.evaluation.scoring import canonical, edit_distance, match_rows


def annotation():
    return {**dict.fromkeys(TEXT_KEYS),'company_name':'Seller','invoice_id':'0001',
            'vehicle_information':dict.fromkeys(VEHICLE_KEYS),
            'items':[{**dict.fromkeys(ITEM_KEYS),'item_code':'A','description':'Part A','quantity':2,'line_total':Decimal('10.00')},
                     {**dict.fromkeys(ITEM_KEYS),'item_code':'B','description':'Part B','quantity':1,'line_total':Decimal('20.00')}],
            'totals':{'subtotal':Decimal('30.00'),'taxes':[],'total_payment':Decimal('30.00')}}


class EvaluationTests(unittest.TestCase):
    def score(self,prediction):
        scorer=Scorer()
        bad=scorer.invoice({'annotation':annotation()},prediction)
        return scorer,bad

    def test_decimal_equivalence_and_identifier_significance(self):
        self.assertTrue(same(Decimal('130.00'),130))
        self.assertFalse(same('0001','1'))
        self.assertFalse(same('130',130))
        self.assertTrue(same('A\n  B','A B'))
        self.assertFalse(same(True,1))

    def test_reordered_items_are_matched_once(self):
        prediction=annotation();prediction['items'].reverse()
        scorer,bad=self.score(prediction)
        self.assertFalse(bad)
        self.assertEqual(scorer.counts['items_rows']['correct'],2)

    def test_missing_item_is_not_scored_as_wrong_remaining_item(self):
        prediction=annotation();prediction['items']=prediction['items'][1:]
        scorer,bad=self.score(prediction)
        self.assertTrue(bad)
        self.assertEqual(scorer.counts['items_rows']['correct'],1)
        self.assertEqual(scorer.counts['items_rows']['missing'],1)
        self.assertEqual(scorer.counts['items[].line_total']['missing'],1)

    def test_extra_item_is_counted(self):
        prediction=annotation();prediction['items'].append(copy.deepcopy(prediction['items'][0]))
        scorer,bad=self.score(prediction)
        self.assertTrue(bad)
        self.assertEqual(scorer.counts['items_rows']['extra'],1)
        self.assertEqual(scorer.counts['items[].unit_price']['missing'],0)

    def test_wrong_missing_extra_and_confirmed_nulls(self):
        prediction=annotation();prediction['invoice_id']='1';prediction['company_name']=None
        prediction['ship_to']='Invented';prediction['bill_to']='Unknown'
        scorer,bad=self.score(prediction)
        self.assertTrue(bad)
        self.assertEqual(scorer.counts['invoice_id']['wrong'],1)
        self.assertEqual(scorer.counts['company_name']['missing'],1)
        self.assertEqual(scorer.counts['ship_to']['extra'],1)
        self.assertEqual(scorer.counts['bill_to']['extra'],1)

    def test_saved_reports_and_failed_invoices_remain_in_denominator(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);truth=root/'labels.jsonl';pred=root/'raw';pred.mkdir()
            images=[{'file':'data/test/id/page_001.jpg','sha256':'a'*64}]
            records=[{'label_format':'invoice-v3','document_id':doc,'source':doc,'images':images,'annotation':annotation()} for doc in ('id1','id2')]
            truth.write_text(''.join(dumps(r)+'\n' for r in records),encoding='utf-8')
            prediction={'label_format':'invoice-v3','document_id':'id1','images':images,'images_sha256':images_digest(images),'status':'ok','extraction':annotation()}
            (pred/'id1.json').write_text(dumps(prediction),encoding='utf-8')
            scorer=evaluate(truth,pred,root/'report')
            self.assertEqual(scorer.counts['schema_valid']['correct'],1)
            self.assertEqual(scorer.counts['schema_valid']['wrong'],1)
            self.assertEqual(scorer.counts['invoice_id']['missing'],1)
            with (root/'report/metrics.csv').open(encoding='utf-8-sig') as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual(next(r for r in rows if r['field']=='invoice_all_fields')['exact_match_pct'],'50.00')
            self.assertTrue((root/'report/errors.csv').is_file())

    def test_wrong_image_order_is_a_failed_invoice(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);truth=root/'labels.jsonl';pred=root/'raw';pred.mkdir()
            images=[{'file':'page_001.jpg','sha256':'a'*64},{'file':'page_002.jpg','sha256':'b'*64}]
            record={'label_format':'invoice-v3','document_id':'id','images':images,'annotation':annotation()}
            truth.write_text(dumps(record)+'\n',encoding='utf-8')
            reversed_images=list(reversed(images))
            result={'label_format':'invoice-v3','document_id':'id','images':reversed_images,
                    'images_sha256':images_digest(reversed_images),'status':'ok','extraction':annotation()}
            (pred/'id.json').write_text(dumps(result),encoding='utf-8')
            scorer=evaluate(truth,pred,root/'report')
            self.assertEqual(scorer.counts['invoice_all_fields']['wrong'],1)
            self.assertEqual(scorer.counts['invoice_id']['missing'],1)

    def test_duplicate_predictions_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);truth=root/'labels.jsonl';pred=root/'raw';pred.mkdir()
            truth.write_text(dumps({'label_format':'invoice-v3','document_id':'id'})+'\n',encoding='utf-8')
            for index in (1,2):
                (pred/f'{index}.json').write_text(dumps({'label_format':'invoice-v3','document_id':'id'}),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'Multiple predictions'):
                evaluate(truth,pred,root/'report')


class FieldScoringTests(unittest.TestCase):
    def test_text_normalization_keeps_significant_content(self):
        self.assertTrue(same(" ACME\u2019S\n  LTD ", "acme's ltd", 'company_name'))
        self.assertTrue(same('Cafe\u0301', 'CAF\u00c9', 'items[].description'))
        self.assertFalse(same('Part 123', 'Part 124', 'items[].description'))
        self.assertFalse(same('DO NOT REPAIR', 'DO REPAIR', 'items[].description'))
        self.assertFalse(same('Seller LTD', 'Seller', 'company_name'))
        self.assertFalse(same('Caf\u00e9', 'Cafe', 'company_name'))

    def test_identifiers_are_not_fuzzy_or_punctuation_stripped(self):
        self.assertFalse(same('001-A', '1-A', 'invoice_id'))
        self.assertFalse(same('001-A', '001A', 'invoice_id'))
        self.assertFalse(same('ABC', 'abc', 'items[].item_code'))

    def test_exact_decimals_have_no_tolerance_or_string_coercion(self):
        self.assertTrue(same(Decimal('1044.3600'), Decimal('1044.36'), 'items[].line_total'))
        self.assertFalse(same(Decimal('1044.36'), Decimal('1044.35'), 'items[].line_total'))
        self.assertFalse(same('1044.36', Decimal('1044.36'), 'items[].line_total'))
        self.assertFalse(same(None, 0, 'items[].line_total'))

    def test_cer_counts_missing_and_extra_characters_but_never_passes_them(self):
        scorer = Scorer()
        self.assertTrue(scorer.cell('items[].description', 'items[0].description', 'part', 'parts'))
        self.assertEqual(scorer.text_counts['items[].description']['edits'], 1)
        self.assertEqual(scorer.text_counts['items[].description']['reference_chars'], 4)
        self.assertEqual(scorer.counts['items[].description']['wrong'], 1)
        self.assertEqual(edit_distance('abc', ''), 3)
        self.assertEqual(edit_distance('abc', 'axc'), 1)

    def test_nulls_do_not_inflate_field_exact_match(self):
        scorer = Scorer()
        self.assertFalse(scorer.cell('company_name', 'company_name', None, None))
        self.assertEqual(scorer.counts['company_name']['null_correct'], 1)
        self.assertEqual(scorer.counts['company_name']['correct'], 0)
        self.assertTrue(scorer.cell('ship_to', 'ship_to', None, 'Invented'))
        self.assertEqual(scorer.counts['ship_to']['extra'], 1)
        with tempfile.TemporaryDirectory() as name:
            scorer.write(Path(name))
            with (Path(name)/'metrics.csv').open(encoding='utf-8-sig') as stream:
                rows = {r['field']:r for r in csv.DictReader(stream)}
            self.assertEqual(rows['company_name']['evaluated'], '0')
            self.assertEqual(rows['company_name']['exact_match_pct'], '')
            self.assertEqual(rows['all_fields']['exact_match_pct'], '0.00')

    def test_duplicate_rows_are_not_matched_to_maximize_amount_accuracy(self):
        expected = [{'item_code':'A', 'description':'Part', 'line_total':1},
                    {'item_code':'A', 'description':'Part', 'line_total':2}]
        predicted = list(reversed(expected))
        self.assertEqual(match_rows(expected, predicted, ('item_code','description')), [(0,0),(1,1)])

    def test_csv_cer_is_weighted_by_reference_characters(self):
        scorer = Scorer()
        scorer.cell('company_name', 'company_name', 'abcd', 'abxd')
        scorer.cell('company_name', 'company_name', 'abcdef', 'abcdef')
        with tempfile.TemporaryDirectory() as name:
            scorer.write(Path(name))
            with (Path(name)/'metrics.csv').open(encoding='utf-8-sig') as stream:
                rows = {r['field']:r for r in csv.DictReader(stream)}
            self.assertEqual(rows['company_name']['cer_pct'], '10.00')
            self.assertEqual(rows['company_name']['exact_match_pct'], '50.00')
            self.assertEqual(rows['company_name']['group'], 'text')
            self.assertEqual(rows['invoice_id']['cer_pct'], '')

if __name__=='__main__':unittest.main()
