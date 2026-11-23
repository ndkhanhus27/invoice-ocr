"""Field-aware invoice scoring. No inference, file loading or label mutation."""
from __future__ import annotations
from collections import Counter, defaultdict
import csv
from decimal import Decimal
import re
import unicodedata
from ..storage.files import dumps_exact as dumps
from ..contracts.invoice import TEXT_KEYS, VEHICLE_KEYS, ITEM_KEYS, TOTAL_KEYS

MISSING = object()
TEXT_FIELDS = frozenset({
    'company_name', 'bill_to', 'ship_to', 'vehicle_information.make',
    'vehicle_information.model', 'items[].description', 'totals.taxes[].label',
})
NUMERIC_FIELDS = frozenset({
    'vehicle_information.year', 'vehicle_information.odometer',
    'items[].quantity', 'items[].unit_price', 'items[].line_total',
    'totals.subtotal', 'totals.total_payment',
    'totals.taxes[].rate', 'totals.taxes[].amount',
})
PUNCTUATION = str.maketrans({
    '\u2018': "'", '\u2019': "'", '\u02bc': "'",
    '\u201c': '"', '\u201d': '"',
    '\u2010': '-', '\u2011': '-', '\u2012': '-', '\u2013': '-',
    '\u2014': '-', '\u2212': '-',
})


def field_key(field):
    return re.sub(r'\[[^]]*\]', '[]', field)


def field_group(field):
    field = field_key(field)
    if field in NUMERIC_FIELDS:
        return 'number'
    return 'text' if field in TEXT_FIELDS else 'identifier'


def canonical(value, field=None):
    if not isinstance(value, str):
        return value
    value = unicodedata.normalize('NFC', value)
    if field is not None and field_group(field) == 'text':
        value = value.translate(PUNCTUATION).casefold()
    return ' '.join(value.split())


def same(a, b, field=None):
    if a is MISSING or b is MISSING:
        return False
    # No bool/float/string coercion into exact monetary values.
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    numeric = (int, Decimal)
    if isinstance(a, numeric) and isinstance(b, numeric):
        return Decimal(a).is_finite() and Decimal(b).is_finite() and Decimal(a) == Decimal(b)
    if field is not None and field_group(field) == 'number':
        return a is None and b is None
    return type(a) is type(b) and canonical(a, field) == canonical(b, field)


def edit_distance(reference, prediction):
    """Levenshtein edits with linear memory; no similarity threshold."""
    if len(reference) < len(prediction):
        reference, prediction = prediction, reference
    previous = list(range(len(prediction) + 1))
    for i, left in enumerate(reference, 1):
        current = [i]
        for j, right in enumerate(prediction, 1):
            current.append(min(current[-1]+1, previous[j]+1,
                               previous[j-1]+(left != right)))
        previous = current
    return previous[-1]


def match_rows(expected, predicted, anchors, field='items'):
    """One-to-one exact anchors, then remaining printed order. Never use money.

    Duplicate codes are disambiguated with textual anchors, then stable order.
    Approximate text never turns a mismatching field into a correct field.
    """
    unused = set(range(len(predicted)))
    matched = {}
    for key in anchors:
        for i, row in enumerate(expected):
            if i in matched or row.get(key) is None:
                continue
            candidates = [j for j in sorted(unused)
                          if same(row[key], predicted[j].get(key), field+'[].'+key)]
            if candidates:
                def anchor_score(j):
                    return sum(row.get(k) is not None and
                               same(row[k], predicted[j].get(k), field+'[].'+k)
                               for k in anchors)
                j = max(candidates, key=anchor_score)
                matched[i] = j
                unused.remove(j)
    remaining = [i for i in range(len(expected)) if i not in matched]
    for i, j in zip(remaining, sorted(unused)):
        matched[i] = j
        unused.remove(j)
    return ([(i, matched.get(i)) for i in range(len(expected))]
            + [(None, j) for j in sorted(unused)])


class Scorer:
    def __init__(self):
        self.counts = defaultdict(Counter)
        self.text_counts = defaultdict(Counter)
        self.errors = []
        self.context = {}
        for field in [*TEXT_KEYS, *('vehicle_information.'+k for k in VEHICLE_KEYS),
                      *('items[].'+k for k in ITEM_KEYS), *('totals.'+k for k in TOTAL_KEYS),
                      *('totals.taxes[].'+k for k in ('label', 'rate', 'amount'))]:
            self.counts[field]

    def error(self, field, status, expected, predicted, note='', text_score=None):
        text_score = text_score or {}
        chars = text_score.get('reference_chars', 0)
        self.errors.append({**self.context, 'field':field, 'status':status,
            'expected':'<missing>' if expected is MISSING else dumps(expected),
            'predicted':'<missing>' if predicted is MISSING else dumps(predicted),
            'note':note, 'cer_edits':text_score.get('edits', ''),
            'cer_reference_chars':text_score.get('reference_chars', ''),
            'cer_pct':format(Decimal(text_score['edits'])*100/chars, '.2f') if chars else ''})

    def cell(self, field, path, expected, predicted):
        if expected is None and predicted is None:
            self.counts[field]['null_correct'] += 1
            return False
        if ((expected is MISSING and predicted is None)
                or (expected is None and predicted is MISSING)):
            self.counts[field]['ignored'] += 1
            return False
        if same(expected, predicted, field):
            status = 'correct'
        elif predicted is MISSING or predicted is None:
            status = 'missing'
        elif expected is MISSING or expected is None:
            status = 'extra'
        else:
            status = 'wrong'
        self.counts[field][status] += 1
        text_score = None
        if field_group(field) == 'text' and (isinstance(expected, str) or isinstance(predicted, str)):
            reference = canonical(expected, field) if isinstance(expected, str) else ''
            prediction = canonical(predicted, field) if isinstance(predicted, str) else ''
            text_score = {'edits':edit_distance(reference, prediction), 'reference_chars':len(reference)}
            self.text_counts[field].update(text_score)
        if status != 'correct':
            self.error(path, status, expected, predicted, text_score=text_score)
        return status != 'correct'

    def rows(self, field, expected, predicted, keys, anchors):
        bad = False
        for i, j in match_rows(expected, predicted, anchors, field):
            status = 'extra' if i is None else 'missing' if j is None else None
            row_bad = False
            for key in keys:
                path = f'{field}[{i if i is not None else "extra:"+str(j)}].{key}'
                a = expected[i][key] if i is not None else MISSING
                b = predicted[j][key] if j is not None else MISSING
                row_bad = self.cell(field+'[].'+key, path, a, b) or row_bad
            if status is None:
                status = 'wrong' if row_bad else 'correct'
            self.counts[field+'_rows'][status] += 1
            if i is None or j is None:
                self.error(field, status, expected[i] if i is not None else MISSING,
                           predicted[j] if j is not None else MISSING)
            bad = bad or status != 'correct'
        return bad

    def invoice(self, truth, prediction):
        expected = truth['annotation']
        bad = False
        for key in TEXT_KEYS:
            bad = self.cell(key, key, expected[key], prediction.get(key, MISSING)) or bad
        vehicle = prediction.get('vehicle_information', {})
        for key in VEHICLE_KEYS:
            field = 'vehicle_information.'+key
            bad = self.cell(field, field, expected['vehicle_information'][key], vehicle.get(key, MISSING)) or bad
        totals = prediction.get('totals', {})
        for key in TOTAL_KEYS:
            field = 'totals.'+key
            bad = self.cell(field, field, expected['totals'][key], totals.get(key, MISSING)) or bad
        bad = self.rows('items', expected['items'], prediction.get('items', []), ITEM_KEYS,
                        ('item_code', 'description')) or bad
        bad = self.rows('totals.taxes', expected['totals']['taxes'], totals.get('taxes', []),
                        ('label', 'rate', 'amount'), ('label',)) or bad
        return bad

    def write(self, directory):
        directory.mkdir(parents=True, exist_ok=True)
        fields = ['field', 'group', 'correct', 'wrong', 'missing', 'extra', 'null_correct',
                  'ignored', 'evaluated', 'exact_match_pct', 'cer_edits', 'cer_reference_chars', 'cer_pct']
        total = Counter()
        total_text = Counter()
        summaries = {'items_rows', 'totals.taxes_rows', 'invoice_all_fields', 'schema_valid'}
        for field, count in self.counts.items():
            if field not in summaries:
                total.update(count)
                total_text.update(self.text_counts[field])
        with (directory/'metrics.csv').open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for field, count in [*self.counts.items(), ('all_fields', total)]:
                n = sum(count[k] for k in ('correct', 'wrong', 'missing', 'extra'))
                text = total_text if field == 'all_fields' else self.text_counts[field]
                chars = text['reference_chars']
                is_text = field in TEXT_FIELDS or field == 'all_fields'
                writer.writerow({'field':field,
                    'group':'summary' if field in summaries or field == 'all_fields' else field_group(field),
                    **{k:count[k] for k in ('correct', 'wrong', 'missing', 'extra', 'null_correct', 'ignored')},
                    'evaluated':n, 'exact_match_pct':format(Decimal(count['correct'])*100/n, '.2f') if n else '',
                    'cer_edits':text['edits'] if is_text else '',
                    'cer_reference_chars':chars if is_text else '',
                    'cer_pct':format(Decimal(text['edits'])*100/chars, '.2f') if chars else ''})
        with (directory/'errors.csv').open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['document_id', 'source', 'field', 'status',
                'expected', 'predicted', 'note', 'cer_edits', 'cer_reference_chars', 'cer_pct'])
            writer.writeheader()
            writer.writerows(self.errors)
