"""Second-pass verification of invoice labels, using shared scoring rules."""
from __future__ import annotations
import hashlib
from ..storage.files import dumps_exact, images_digest
from ..contracts.invoice import TEXT_KEYS, VEHICLE_KEYS, ITEM_KEYS, TOTAL_KEYS, validate_invoice
from ..evaluation.scoring import MISSING, same, match_rows


def annotation_hash(annotation):
    return hashlib.sha256(dumps_exact(annotation).encode('utf-8')).hexdigest()


def has_content(value):
    if isinstance(value, dict):
        return any(has_content(v) for v in value.values())
    if isinstance(value, list):
        return any(has_content(v) for v in value)
    return value is not None and value != ''


def compare_annotations(label, check_result):
    """No fuzzy acceptance: exact numbers/IDs, normalized text, one-to-one rows."""
    validate_invoice(label)
    validate_invoice(check_result)
    differences = []
    def cell(field, left, right, checker_field=None):
        if not same(left, right, field):
            differences.append({'field':field, 'checker_field':checker_field or field,
                'label_value':'<missing>' if left is MISSING else dumps_exact(left),
                'checker_value':'<missing>' if right is MISSING else dumps_exact(right)})
    for key in TEXT_KEYS:
        cell(key, label[key], check_result[key])
    for key in VEHICLE_KEYS:
        cell('vehicle_information.'+key, label['vehicle_information'][key], check_result['vehicle_information'][key])
    for key in TOTAL_KEYS:
        cell('totals.'+key, label['totals'][key], check_result['totals'][key])
    for field, left, right, keys, anchors in (
        ('items', label['items'], check_result['items'], ITEM_KEYS, ('item_code','description')),
        ('totals.taxes', label['totals']['taxes'], check_result['totals']['taxes'], ('label','rate','amount'), ('label',)),
    ):
        for i,j in match_rows(left, right, anchors, field):
            if i is None:
                cell(f'{field}[extra:{j}]', MISSING, right[j], f'{field}[{j}]')
            elif j is None:
                cell(f'{field}[{i}]', left[i], MISSING, '<missing>')
            else:
                for key in keys:
                    cell(f'{field}[{i}].{key}', left[i][key], right[j][key], f'{field}[{j}].{key}')
    if not has_content(label) and not has_content(check_result):
        differences.append({'field':'__evidence__','checker_field':'__evidence__',
                            'label_value':'No readable annotation','checker_value':'No readable annotation'})
    return differences


def refresh_verification(record):
    """Recompare after an edit; a human may explicitly approve a disagreement."""
    verification = record.get('verification')
    if not isinstance(verification, dict):
        return
    allowed = {'unlabeled','pending','agreed','disagreed','error'}
    if verification.get('status') not in allowed:
        raise ValueError('Invalid verification status')
    candidate = verification.get('candidate')
    if candidate is not None:
        differences = compare_annotations(record['annotation'], candidate)
        verification['differences'] = differences
        verification['status'] = 'disagreed' if differences else 'agreed'
        verification['annotation_sha256'] = annotation_hash(record['annotation'])
    if record.get('review',{}).get('action') == 'automatic_agreement' and record.get('review_status') == 'reviewed':
        if (verification.get('status') != 'agreed' or candidate is None
                or verification.get('images_sha256') != images_digest(record['images'])):
            raise ValueError('Automatic approval requires matching annotations and image identity')


def reset_for_verification(record):
    """Preserve the current label, including previous human corrections."""
    record['review_status'] = 'needs_review'
    record['verification'] = {'status':'pending' if has_content(record['annotation']) else 'unlabeled'}
    record['review'] = {**(record.get('review') or {}), 'reviewed_at':None, 'action':'pending_verification'}
    return record
