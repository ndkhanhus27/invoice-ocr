"""The typed business contract is shared by all extraction/labeling backends."""
import copy
from decimal import Decimal
import unittest

from invoice_ocr.contracts.invoice import TEXT_KEYS, VEHICLE_KEYS, parse_invoice, validate_invoice
from invoice_ocr.storage.files import dumps_exact


def invoice():
    return {**dict.fromkeys(TEXT_KEYS), 'invoice_id': '0001', 'purchase_order': '00042',
            'vehicle_information': dict.fromkeys(VEHICLE_KEYS),
            'items': [{'item_code': None, 'description': 'Part', 'quantity': 1,
                       'unit_price': Decimal('12.3456'), 'line_total': Decimal('12.3456')}],
            'totals': {'subtotal': Decimal('12.3456'), 'taxes': [], 'total_payment': Decimal('12.3456')}}


class ContractTests(unittest.TestCase):
    def test_decimal_and_leading_zero_ids_are_preserved(self):
        result = parse_invoice(dumps_exact(invoice()))
        self.assertEqual(result['totals']['total_payment'], Decimal('12.3456'))
        self.assertEqual(result['purchase_order'], '00042')

    def test_missing_extra_and_invalid_fields_are_rejected(self):
        mutations = [lambda d: d.pop('purchase_order'), lambda d: d.update(unexpected=1),
                     lambda d: d.update(invoice_id=123), lambda d: d.update(date='03/04/2025'),
                     lambda d: d.update(items='not an array'),
                     lambda d: d['items'][0].update(quantity='1'),
                     lambda d: d['totals'].update(total_payment=True),
                     lambda d: d['totals'].update(total_payment=Decimal('1.12345'))]
        for mutate in mutations:
            data = copy.deepcopy(invoice())
            mutate(data)
            with self.subTest(data=data), self.assertRaises(ValueError):
                validate_invoice(data)
