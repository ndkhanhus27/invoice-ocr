"""Typed business contract shared by extraction, labeling and persistence."""
from __future__ import annotations

import json
from decimal import Decimal
from datetime import date
import re

TEXT_KEYS = ("company_name", "bill_to", "ship_to", "invoice_id", "date", "purchase_order", "currency")

VEHICLE_KEYS = ("vin", "licence_plate", "year", "make", "model", "unit", "odometer", "odometer_unit", "driver_licence")

ITEM_KEYS = ("item_code", "description", "quantity", "unit_price", "line_total")

TOTAL_KEYS = ("subtotal", "total_payment")

NORMALIZATION_VERSION = "2.0"

NUMERIC_ITEM_KEYS = ("quantity", "unit_price", "line_total")

def validate_currency(value):
    if value is not None and (not isinstance(value, str) or not re.fullmatch(r'[A-Z]{3}', value)):
        raise ValueError('currency must be a three-letter uppercase code such as CAD/USD/VND, or null')


def numeric_value(value, precision, scale, field):
    """Validate a JSON number decoded exactly; never accept bool/float/string."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, Decimal)):
        raise ValueError(f"{field} must be a number or null (decode decimals with json.loads(parse_float=Decimal))")
    number = Decimal(value)
    if not number.is_finite():
        raise ValueError(f"{field} must be finite")
    fractional = max(0, -number.as_tuple().exponent)
    integer = max(0, number.adjusted() + 1) if number else 0
    if fractional > scale or integer > precision - scale:
        raise ValueError(f"{field} exceeds NUMERIC({precision},{scale})")
    return number


def validate_invoice(data):
    """Validate the shared invoice-v3 business contract without coercion."""
    def shape(obj, keys, field):
        if not isinstance(obj, dict) or set(obj) != set(keys):
            raise ValueError(f"{field}: missing or unexpected keys")
    def texts(obj, keys, field):
        for key in keys:
            if obj[key] is not None and not isinstance(obj[key], str):
                raise ValueError(f"{field}.{key} must be a string or null")
    shape(data, (*TEXT_KEYS, 'vehicle_information', 'items', 'totals'), 'invoice')
    texts(data, TEXT_KEYS, 'invoice')
    validate_currency(data['currency'])
    if data['date'] is not None:
        if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', data['date']) or date.fromisoformat(data['date']).isoformat() != data['date']:
            raise ValueError('date must be an ISO date YYYY-MM-DD')
    vehicle = data['vehicle_information']
    shape(vehicle, VEHICLE_KEYS, 'vehicle_information')
    texts(vehicle, [k for k in VEHICLE_KEYS if k not in {'year', 'odometer'}], 'vehicle_information')
    year = numeric_value(vehicle['year'], 5, 0, 'vehicle_information.year')
    if year is not None and not -32768 <= year <= 32767:
        raise ValueError('vehicle_information.year exceeds SMALLINT')
    numeric_value(vehicle['odometer'], 18, 3, 'vehicle_information.odometer')
    if vehicle['odometer_unit'] not in (None, 'km', 'mi'):
        raise ValueError('vehicle_information.odometer_unit must be km, mi or null')
    if not isinstance(data['items'], list):
        raise ValueError('items must be an array')
    for index, item in enumerate(data['items']):
        shape(item, ITEM_KEYS, f'items[{index}]')
        texts(item, ('item_code', 'description'), f'items[{index}]')
        for key in NUMERIC_ITEM_KEYS:
            numeric_value(item[key], 18, 6 if key in {'quantity', 'unit_price'} else 4, f'items[{index}].{key}')
    totals = data['totals']
    shape(totals, (*TOTAL_KEYS, 'taxes'), 'totals')
    for key in TOTAL_KEYS:
        numeric_value(totals[key], 18, 4, 'totals.' + key)
    if not isinstance(totals['taxes'], list):
        raise ValueError('taxes must be an array')
    for index, tax in enumerate(totals['taxes']):
        shape(tax, ('label', 'rate', 'amount'), f'taxes[{index}]')
        texts(tax, ('label',), f'taxes[{index}]')
        rate = numeric_value(tax['rate'], 12, 8, f'taxes[{index}].rate')
        if rate is not None and rate < 0:
            raise ValueError('tax rate cannot be negative')
        numeric_value(tax['amount'], 18, 4, f'taxes[{index}].amount')
    return data


def parse_invoice(text):
    """Decode the model's typed answer exactly and validate, without coercion."""
    cleaned = text.strip()
    if cleaned.startswith('```'):
        cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned, count=1)
        cleaned = re.sub(r'\s*```$', '', cleaned, count=1)
    def invalid_constant(value):
        raise ValueError('Non-finite JSON number: ' + value)
    data = json.loads(cleaned, parse_float=Decimal, parse_constant=invalid_constant)
    return validate_invoice(data)


def invoice_json_schema():
    """Structured output shape shares field names with the project validator."""
    def obj(properties):
        return {'type':'object','properties':properties,'required':list(properties),'additionalProperties':False}
    def scalar(kind='string', **kwargs):
        return {'type':[kind,'null'], **kwargs}
    vehicle = {k:scalar() for k in VEHICLE_KEYS}
    vehicle.update(year=scalar('integer', minimum=-32768, maximum=32767),
                   odometer=scalar('number'), odometer_unit=scalar(enum=['km','mi',None]))
    item = {k:scalar('number' if k in NUMERIC_ITEM_KEYS else 'string') for k in ITEM_KEYS}
    properties = {k:scalar() for k in TEXT_KEYS}
    properties['currency'] = scalar(pattern='^[A-Z]{3}$')
    properties['date'] = scalar(pattern=r'^\d{4}-\d{2}-\d{2}$')
    properties.update(vehicle_information=obj(vehicle), items={'type':'array','items':obj(item)},
        totals=obj({'subtotal':scalar('number'),'total_payment':scalar('number'),
                    'taxes':{'type':'array','items':obj({'label':scalar(), 'rate':scalar('number',minimum=0),'amount':scalar('number')})}}))
    return obj(properties)


