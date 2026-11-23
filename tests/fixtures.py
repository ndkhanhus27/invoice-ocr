"""Shared test invoices; string-valued fixture is used only to test rejection."""
from invoice_ocr.contracts import invoice as schema

def invoice():
    return {
        **dict.fromkeys(schema.TEXT_KEYS),
        "vehicle_information": dict.fromkeys(schema.VEHICLE_KEYS),
        "items": [{**dict.fromkeys(schema.ITEM_KEYS), "description": "WINDSHIELD", "line_total": "529.00"}],
        "totals": {"subtotal": "529.00", "taxes": [
            {"label": "HST", "rate": "13%", "amount": "68.77"}], "total_payment": "597.77"},
    }

def typed_invoice():
    data = invoice()
    data['items'][0]['line_total'] = 529
    data['totals'] = {'subtotal': 529, 'taxes': [{'label': 'HST', 'rate': 0.13, 'amount': 68.77}], 'total_payment': 597.77}
    return data
