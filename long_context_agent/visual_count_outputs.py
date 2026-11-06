"""Whole-scope scalar recovery for visual-count audit tools.

Only use this when every required source was actually supplied in one request.
It does not recover missing observations or turn a page answer into a total.
"""
import json
import re

from .multimodal_page_harness_v2 import json_values
from .visual_count_memory_v21 import integer


def explicit_global_answer(raw):
    text = str(raw or '').strip()
    if text.startswith('```'):
        text = re.sub(r'^```(?:json)?\s*|\s*```$', '', text)
    try:
        root = json.loads(text)
    except ValueError:
        root = next(json_values(text), None)
    scalar = integer(root)
    if scalar is not None:
        return scalar
    if not isinstance(root, dict) or any(k in root for k in ['id', 'page_id', 'source_id']):
        return None
    for key in ['answer', 'combined_total', 'grand_total', 'total_count']:
        if key in root and (number := integer(root[key])) is not None:
            return number
    if all(k in root and integer(root[k]) is not None for k in ['figure_total', 'table_total']):
        return integer(root['figure_total']) + integer(root['table_total'])
    totals = root.get('totals')
    fields = ('figures', 'tables')
    if isinstance(totals, dict) and all(k in totals and integer(totals[k]) is not None for k in fields):
        return sum(integer(totals[k]) for k in fields)
    return None
