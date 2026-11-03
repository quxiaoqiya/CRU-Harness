#!/usr/bin/env python3
"""Audit a frozen context comparison; labels stay unopened unless explicitly scored.

Default audits hash inference inputs without parsing their text and never open or
hash labels. --verify-scores requires every frozen prediction and journal, passes
the execution checks first, then independently executes only the two synth
functions extracted by AST from the pinned official scorer. No model is called.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import time

from scripts.benchmark_context_holdout import atomic_json, digest_file, runtime_identity

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / 'docs/context_holdout_cohort3_v3_20261101'
OFFICIAL_COMMIT = '0bb7eabe839218fee7fe8d007f41cfc2fd3ae24c'


def read_json(path: Path):
    return json.loads(path.read_text())


def is_count(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def audit_ledger(ledger: list, budget: dict) -> dict:
    """Recompute sequential admission and charging, independent of summary flags."""
    violations, warnings, conservation = [], [], []
    prompt_total = completion_total = charged_total = native = unknown = 0
    statuses, margins = Counter(), Counter()
    if len(ledger) > budget['max_calls']:
        violations.append('physical_request_limit')
    for index, entry in enumerate(ledger, 1):
        prefix = f'request_{index}:'
        statuses[entry.get('status', 'missing')] += 1
        native += bool(entry.get('native_tools'))
        if entry.get('physical_request') != index:
            violations.append(prefix + 'nonsequential_identity')
        values = [entry.get(k) for k in ('admission_prompt_tokens', 'requested_output_tokens', 'charged_tokens')]
        if not all(is_count(x) for x in values):
            violations.append(prefix + 'invalid_token_counts')
            continue
        prompt, output, charged = values
        reservation = prompt + output
        remaining = budget['total_tokens'] - charged_total
        if not 0 < output <= budget['max_output_tokens']:
            violations.append(prefix + 'output_limit')
        if reservation > budget['context_tokens']:
            violations.append(prefix + 'context_limit')
        if reservation > remaining:
            violations.append(prefix + 'cumulative_admission_limit')
        usage = entry.get('usage') or {}
        known = all(is_count(usage.get(k)) for k in ('prompt_tokens', 'completion_tokens'))
        if entry.get('usage_known', False) != known:
            violations.append(prefix + 'usage_known_flag')
        if known:
            actual_prompt, completion = usage['prompt_tokens'], usage['completion_tokens']
            prompt_total += actual_prompt
            completion_total += completion
            margins[prompt - actual_prompt] += 1
            if actual_prompt > prompt or completion > output:
                violations.append(prefix + 'provider_exceeded_reservation')
            if charged != actual_prompt + completion:
                violations.append(prefix + 'reported_usage_charge_mismatch')
            if 'total_tokens' in usage and usage['total_tokens'] != actual_prompt + completion:
                violations.append(prefix + 'provider_total_mismatch')
        else:
            unknown += 1
            if charged != reservation:
                violations.append(prefix + 'unknown_usage_not_fully_reserved')
            warnings.append(prefix + 'unknown_usage_conservatively_charged')
        if entry.get('status') not in {'ok', 'error', 'inflight'}:
            violations.append(prefix + 'invalid_status')
        if entry.get('error_type') == 'BudgetIntegrityError':
            violations.append(prefix + 'provider_budget_integrity_error')
        if entry.get('native_tools') and not entry.get('tools_sha256'):
            violations.append(prefix + 'native_schema_hash_missing')
        if not isinstance(entry.get('prompt_sha256'), str) or len(entry['prompt_sha256']) != 64:
            violations.append(prefix + 'prompt_hash_missing')
        charged_total += charged
        if charged_total > budget['total_tokens']:
            violations.append(prefix + 'cumulative_charge_limit')
        conservation.append({'request': index, 'status': entry.get('status'),
                             'native_tools': bool(entry.get('native_tools')),
                             'reserved_tokens': reservation, 'charged_tokens': charged,
                             'remaining_tokens_before': remaining,
                             'remaining_tokens_after': budget['total_tokens'] - charged_total})
    return {'physical_requests': len(ledger), 'prompt_tokens': prompt_total,
            'completion_tokens': completion_total, 'charged_tokens': charged_total,
            'native_tool_requests': native, 'unknown_usage_requests': unknown,
            'statuses': dict(statuses), 'admission_minus_reported_prompt': dict(margins),
            'conservation': conservation, 'violations': violations, 'warnings': warnings}


def source_checks(output: Path, frozen: dict, root: Path) -> dict:
    missing, snapshots, current = [], [], []
    for name, expected in frozen['code_sha256'].items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            missing.append(name + ':unsafe_relative_path')
            continue
        snapshot = output / 'frozen_source' / relative
        if not snapshot.is_file():
            missing.append(name)
        elif digest_file(snapshot) != expected:
            snapshots.append(name)
        if not (root / relative).is_file() or digest_file(root / relative) != expected:
            current.append(name)
    return {'declared_files': len(frozen['code_sha256']), 'missing_snapshot_files': missing,
            'snapshot_hash_mismatches': snapshots, 'current_declared_file_differences': current}


def guard_checks(output: Path, frozen: dict, manifest_path: Path) -> dict:
    profile = output / 'labels_guard.sb'
    raw = profile.read_text()
    labels = str((manifest_path.parent / 'labels.json').resolve())
    probe = str(Path(frozen['label_guard_probe']).resolve())
    denied = lambda path: '(deny file-read* (literal ' + json.dumps(path) + '))' in raw
    verification = read_json(output / 'guard_verification.json') if (output / 'guard_verification.json').exists() else {}
    return {'profile_sha256': digest_file(profile),
            'profile_matches_freeze': digest_file(profile) == frozen['label_guard_profile_sha256'],
            'labels_literal_denied': denied(labels), 'probe_literal_denied': denied(probe),
            'probe_read_denied': verification.get('probe_read_denied') is True,
            'guard_verified': verification.get('verified') is True,
            'labels_not_opened_by_runner': frozen.get('labels_opened_by_runner') is False and verification.get('labels_not_opened') is True,
            'verification_profile_matches': verification.get('profile_sha256') == frozen['label_guard_profile_sha256'],
            'verification_at_unix': verification.get('time'),
            'labels_file_opened_by_auditor': False, 'labels_file_hashed_by_auditor': False,
            'proof_limit': 'Probe denial and frozen profile/launcher establish a guard check, not a complete OS file-access trace.'}


def load_official_scorer(path: Path):
    """Execute pinned scorer functions without importing its unrelated ML stack."""
    import dateutil.parser
    wanted = {'synth_attempt_answer_parse', 'synth_process_response'}
    selected = [node for node in ast.parse(path.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name in wanted]
    if {node.name for node in selected} != wanted:
        raise ValueError('Pinned official synth functions are missing')
    namespace = {'ast': ast, 'datetime': datetime, 'dateutil': dateutil}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['synth_process_response']


def _span_bounds(value):
    match = re.fullmatch(r'source_chars (\d+):(\d+)', str(value))
    return tuple(map(int, match.groups())) if match else None


def _canonical_json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def audit_body_projection(binding: dict, source_record, context: str, *, source_sha256: str | None = None) -> list[str]:
    """Independently locate the body token, including repeated-header decoys."""
    errors = []
    sha = source_sha256 if source_sha256 is not None else hashlib.sha256(context.encode()).hexdigest()
    if (binding.get('record_id') != source_record.record_id
            or binding.get('source_span') != source_record.source_span
            or binding.get('source_sha256') != sha):
        errors.append('projection_full_record_identity_mismatch')
    body = source_record.fields.get('body')
    expected = None
    if isinstance(body, str) and body:
        if source_record.text.lstrip().startswith('{'):
            for match in re.finditer(r'"(?:body|instance|text|content|message)"\s*:\s*("(?:[^"\\]|\\.)*")', source_record.text, re.I):
                token = match.group(1)
                try:
                    decoded = json.loads(token)
                except ValueError:
                    continue
                if decoded == body and token[1:-1] == body:
                    start = source_record.start + match.start(1) + 1
                    expected = (start, start + len(body))
                    break
        else:
            match = re.search(r'(?:^|\|\|)[ \t]*(?:body|instance|text|content|message)[ \t]*:[ \t]*', source_record.text, re.I)
            if match:
                suffix = source_record.text[match.end():]
                start = source_record.start + match.end() + len(suffix) - len(suffix.lstrip())
                expected = (start, start + len(body))
    bounds = _span_bounds(binding.get('body_source_span'))
    if (expected is None or bounds != expected or not source_record.start <= expected[0] < expected[1] <= source_record.end
            or context[expected[0]:expected[1]] != body):
        errors.append('projection_body_source_range_mismatch')
    elif binding.get('body_sha256') != hashlib.sha256(body.encode()).hexdigest():
        errors.append('projection_body_hash_mismatch')
    if bounds and any(key in binding and binding[key] != value for key, value in zip(('body_start', 'body_end'), bounds)):
        errors.append('projection_body_offset_fields_mismatch')
    return errors


def _replay_record_reply(raw: str, aliases: list[str], bindings: dict, field: str, labels: list[str]) -> dict:
    """Independent strict replay; never discard invalid positions then enumerate."""
    reply = None
    candidates = [str(raw or '').strip()]
    if '{' in candidates[0] and '}' in candidates[0]:
        candidates.append(candidates[0][candidates[0].find('{'):candidates[0].rfind('}') + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        if isinstance(value, dict):
            reply = value
            break
    items, rejected = [], []
    result = {'format': 'unknown', 'batch_rejected': False, 'items': items, 'rejected': rejected}
    reject = lambda reason, alias=None, index=None: rejected.append((reason, alias, index))
    if reply is None:
        # The controller's JSON recovery turns malformed/non-object output into
        # a parse_error object before calling its decoder.
        reject('missing_values_or_records')
        result['batch_rejected'] = True
        return result

    def add(alias, value, index, *, compact, item=None):
        binding = bindings[alias]
        if item is not None:
            span = item.get('source_span')
            if 'source_span' in item:
                match = re.fullmatch(r'(?:source_chars\s*)?(\d+)\s*:\s*(\d+)', str(span).strip())
                if not match or f'source_chars {int(match.group(1))}:{int(match.group(2))}' != binding.get('source_span'):
                    reject('source_span_mismatch', alias, index)
                    return
            for key, reason in (('record_id', 'record_identity_mismatch'), ('source_sha256', 'source_hash_mismatch')):
                if key in item and item[key] != binding.get(key):
                    reject(reason, alias, index)
                    return
        if value is None:
            reject('unknown_value', alias, index)
        elif field == 'label':
            if compact:
                if isinstance(value, bool) or not isinstance(value, int):
                    reject('taxonomy_index_not_integer', alias, index)
                    return
                if not 0 <= value < len(labels):
                    reject('taxonomy_index_out_of_range', alias, index)
                    return
                value = labels[value]
            else:
                matches = [label for label in labels if isinstance(value, str) and label.strip().casefold() == value.strip().casefold()]
                if len(matches) != 1:
                    reject('value_outside_taxonomy', alias, index)
                    return
                value = matches[0]
        elif isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            reject('numeric_value_required', alias, index)
            return
        items.append({'alias': alias, 'record_id': binding.get('record_id'), 'value': value})

    if 'values' in reply and 'records' in reply:
        reject('ambiguous_reply_formats')
        result['batch_rejected'] = True
    elif 'values' in reply:
        result['format'] = 'compact_values_v1'
        values = reply['values']
        if not isinstance(values, list) or len(values) != len(aliases):
            rejected.extend(('vector_length_mismatch', alias, index) for index, alias in enumerate(aliases))
            result['batch_rejected'] = True
        else:
            for index, alias in enumerate(aliases):
                add(alias, values[index], index, compact=True)
    elif isinstance(reply.get('records'), list):
        rows = reply['records']
        has_pairs = any(isinstance(item, list) for item in rows)
        has_dicts = any(isinstance(item, dict) for item in rows)
        if has_pairs and has_dicts:
            reject('mixed_record_formats')
            result['batch_rejected'] = True
            return result
        if has_pairs or not has_dicts:
            result['format'] = 'compact_alias_pairs_v1'
            counts = Counter(item[0] for item in rows if isinstance(item, list) and len(item) == 2 and isinstance(item[0], str))
            for index, item in enumerate(rows):
                if not isinstance(item, list) or len(item) != 2:
                    reject('invalid_alias_pair', index=index)
                    continue
                alias, value = item
                if not isinstance(alias, str):
                    reject('alias_not_string', index=index)
                elif alias not in bindings:
                    reject('unknown_alias', alias, index)
                elif counts[alias] > 1:
                    reject('duplicate_alias', alias, index)
                else:
                    add(alias, value, index, compact=True)
            rejected.extend(('missing_alias', alias, index) for index, alias in enumerate(aliases) if not counts[alias])
            return result
        result['format'] = 'legacy_records_v1'
        counts = Counter(str(item.get('id')) for item in rows if isinstance(item, dict))
        for index, item in enumerate(rows):
            if not isinstance(item, dict):
                reject('invalid_legacy_record', index=index)
                continue
            alias = str(item.get('id', ''))
            if alias not in bindings:
                reject('unknown_alias', alias, index)
            elif counts[alias] > 1:
                reject('duplicate_alias', alias, index)
            else:
                add(alias, item.get('value', item.get('label')), index, compact=False, item=item)
        rejected.extend(('missing_alias', alias, index) for index, alias in enumerate(aliases) if not counts[alias])
    else:
        reject('missing_values_or_records')
        result['batch_rejected'] = True
    return result


def audit_record_protocol(record: dict, context: str, collection, budget: dict | None = None) -> dict:
    """Bind raw vector positions to source identities; old traces remain valid."""
    result = record.get('result') or {}
    plan = result.get('plan') or {}
    field, labels = str(plan.get('field', 'label')), plan.get('labels') or []
    rounds = [x for x in result.get('trace', []) if x.get('action') == 'classify_records' and isinstance(x.get('record_protocol'), dict)]
    errors, known_bindings, expected_values, body_ranges = [], [], {}, set()
    expected_fanout, inferred_ranges, fanout_ids = {}, set(), set()
    fanout_reference_ranges, group_attempts = set(), Counter()
    response_replays, strict_batch_rejections = 0, 0
    by_id = collection.by_id
    source_sha = hashlib.sha256(context.encode()).hexdigest()
    normalize = lambda value: _canonical_json(value.strip().casefold() if field == 'label' and isinstance(value, str) else value)
    for round_index, review in enumerate(rounds):
        prefix = f'protocol_round_{round_index}:'
        protocol = review['record_protocol']
        aliases, selected = review.get('selected_aliases') or [], review.get('selected_record_ids') or []
        bindings = protocol.get('bindings') or {}
        if aliases != [f'r{i}' for i in range(len(selected))] or set(bindings) != set(aliases):
            errors.append(prefix + 'selected_alias_order_mismatch')
        if len(set(selected)) != len(selected) or len(aliases) != len(selected):
            errors.append(prefix + 'selected_record_identity_duplicates_or_length')
        rows, selected_bodies = [], []
        valid = True
        for index, alias in enumerate(aliases):
            binding = bindings.get(alias) or {}
            source_record = by_id.get(selected[index]) if index < len(selected) else None
            if source_record is None or binding.get('record_id') != source_record.record_id or binding.get('alias') != alias:
                errors.append(prefix + 'alias_to_full_record_order_mismatch')
                valid = False
                continue
            errors.extend(prefix + error for error in audit_body_projection(binding, source_record, context, source_sha256=source_sha))
            bounds = _span_bounds(binding.get('body_source_span'))
            if bounds:
                body_ranges.add(bounds)
                selected_bodies.append(binding['body_source_span'])
            rows.append({'id': alias, 'text': source_record.fields.get('body')})
            if (result.get('semantic_coverage') or {}).get('enabled'):
                group_attempts[source_record.fields.get('body')] += 1
            known_bindings.append((binding, protocol.get('format')))
        wire_hash = hashlib.sha256(_canonical_json(rows).encode()).hexdigest()
        if protocol.get('wire_sha256') != wire_hash or any(binding.get('projection_sha256') != wire_hash for binding in bindings.values()):
            errors.append(prefix + 'wire_body_projection_hash_mismatch')
        if ('bindings_sha256' in protocol
                and protocol['bindings_sha256'] != hashlib.sha256(_canonical_json(bindings).encode()).hexdigest()):
            errors.append(prefix + 'local_binding_map_hash_mismatch')
        if review.get('body_exposed_spans') != selected_bodies:
            errors.append(prefix + 'body_exposure_order_mismatch')
        if review.get('source_spans') != [by_id[rid].source_span for rid in selected if rid in by_id]:
            errors.append(prefix + 'full_source_identity_order_mismatch')
        physical = None
        if budget is not None and is_count((review.get('budget_before') or {}).get('calls')):
            physical = budget['max_calls'] - review['budget_before']['calls'] + 1
        elif len(record.get('ledger') or []) in {len(rounds), len(rounds) + 1}:
            physical = round_index + 1 + (len(record['ledger']) - len(rounds))
        if not valid or physical is None or not 1 <= physical <= len(record.get('ledger') or []):
            errors.append(prefix + 'raw_classification_response_unbound')
            continue
        replay = _replay_record_reply(record['ledger'][physical - 1].get('content', ''), aliases, bindings, field, labels)
        response_replays += 1
        strict_batch_rejections += replay['batch_rejected']
        if replay['format'] != protocol.get('format') or replay['batch_rejected'] != protocol.get('batch_rejected'):
            errors.append(prefix + 'decoder_format_or_batch_rejection_mismatch')
        actual_rejections = Counter((e.get('reason'), e.get('alias'), e.get('index')) for e in review.get('fact_events', [])
                                    if e.get('status') in {'rejected', 'unknown'} and 'field' not in e)
        if actual_rejections != Counter(replay['rejected']):
            errors.append(prefix + 'raw_response_rejections_mismatch')
        expected_records = Counter(item['record_id'] for item in replay['items'])
        actual_records = Counter(e.get('record_id') for e in review.get('fact_events', [])
                                 if e.get('status') in {'accepted', 'duplicate', 'conflict', 'unknown'} and e.get('field') == field)
        if actual_records != expected_records:
            errors.append(prefix + 'strict_vector_alias_mapping_or_no_shift_mismatch')
        replay_items = {item['record_id']: item for item in replay['items'] if item['value'] is not None}
        for item in replay['items']:
            if item['value'] is not None:
                expected_values.setdefault((item['record_id'], field), set()).add(normalize(item['value']))
        seen_fanout = set()
        for event in review.get('fanout_events') or []:
            rid, representative_id = event.get('record_id'), event.get('representative_record_id')
            item = replay_items.get(representative_id)
            target, representative = by_id.get(rid), by_id.get(representative_id)
            if ((rid, representative_id) in seen_fanout or rid == representative_id or target is None
                    or representative is None or item is None or event.get('field') != field or field != 'label'
                    or event.get('status') not in {'accepted', 'duplicate', 'conflict'}):
                errors.append(prefix + 'fanout_event_not_bound_to_valid_representative')
                continue
            seen_fanout.add((rid, representative_id))
            target_body, representative_body = target.fields.get('body'), representative.fields.get('body')
            if (not isinstance(target_body, str) or target_body != representative_body
                    or event.get('body_sha256') != hashlib.sha256(target_body.encode()).hexdigest()):
                errors.append(prefix + 'fanout_body_not_exactly_equal')
                continue
            expected_values.setdefault((rid, field), set()).add(normalize(item['value']))
            expected_fanout.setdefault((rid, field), []).append((representative_id, item['value'], bindings[item['alias']], protocol.get('format'), event.get('status')))
            if event.get('status') in {'accepted', 'duplicate'}:
                fanout_ids.add(rid)

    known_projection_signatures = {_canonical_json({**binding, 'protocol': fmt}) for binding, fmt in known_bindings}
    for rid, fields in (result.get('record_facts') or {}).items():
        for name, cell in fields.items():
            for ref in cell.get('evidence') or []:
                provenance = ref.get('provenance') or {}
                projection = provenance.get('projection') if isinstance(provenance, dict) else None
                if provenance.get('stage') == 'exact_body_fanout':
                    representative_id = provenance.get('representative_record_id')
                    representative = by_id.get(representative_id)
                    source_record = by_id.get(rid)
                    candidates = expected_fanout.get((rid, name), [])
                    projection = provenance.get('representative_projection') or {}
                    matching = [(value, binding, fmt, status) for source_id, value, binding, fmt, status in candidates
                                if source_id == representative_id and projection == {**binding, 'protocol': fmt}]
                    if not matching or source_record is None or representative is None:
                        errors.append('fanout_provenance_not_bound_to_raw_response')
                        continue
                    target_body, representative_body = source_record.fields.get('body'), representative.fields.get('body')
                    if (not isinstance(target_body, str) or target_body != representative_body
                            or provenance.get('equality') != 'exact_utf8_body' or provenance.get('metadata_used') is not False
                            or provenance.get('representative_source_span') != representative.source_span):
                        errors.append('fanout_provenance_not_exact_body_only')
                    binding = {'record_id': rid, 'source_span': source_record.source_span,
                               'source_sha256': source_sha, **provenance}
                    errors.extend('fanout_' + error for error in audit_body_projection(binding, source_record, context, source_sha256=source_sha))
                    bounds = _span_bounds(provenance.get('body_source_span'))
                    if bounds:
                        fanout_reference_ranges.add(bounds)
                        if any(status in {'accepted', 'duplicate'} for value, binding, fmt, status in matching):
                            inferred_ranges.add(bounds)
                        if provenance.get('body_characters') != bounds[1] - bounds[0]:
                            errors.append('fanout_body_character_length_mismatch')
                    continue
                if projection is None:
                    if rounds:
                        errors.append('protocol_fact_missing_projection_provenance')
                    continue  # old records-format evidence is intentionally supported
                source_record = by_id.get(rid)
                if not isinstance(projection, dict) or source_record is None:
                    errors.append('evidence_projection_missing_source_record')
                    continue
                errors.extend(audit_body_projection(projection, source_record, context, source_sha256=source_sha))
                if _canonical_json(projection) not in known_projection_signatures:
                    errors.append('evidence_projection_not_bound_to_selected_batch')
            if rounds:
                expected = set(expected_values.get((rid, name), set()))
                source_value = by_id[rid].fields.get(name) if rid in by_id else None
                if source_value is not None:
                    expected.add(normalize(source_value))
                actual = {normalize(value) for value in cell.get('candidates') or []}
                if expected != actual:
                    errors.append('raw_response_value_to_source_candidate_mismatch')
    if any(count > 2 for count in group_attempts.values()):
        errors.append('exact_body_group_classification_attempt_limit')
    if any(review.get('fanout_events') for review in rounds):
        actual_inferred = {_span_bounds(value) for value in result.get('inferred_body_spans') or []}
        if actual_inferred != inferred_ranges:
            errors.append('fanout_inferred_body_coverage_mismatch')
        summary = result.get('semantic_coverage') or {}
        if (summary.get('enabled') is not True or summary.get('equality') != 'exact_utf8_body'
                or summary.get('normalization') != 'none' or summary.get('derived_facts_are_model_exposure') is not False
                or summary.get('transferred_records') != len(fanout_ids)
                or summary.get('direct_model_classified_records') != len({rid for review in rounds for rid in review.get('selected_record_ids') or []})):
            errors.append('fanout_summary_mismatch')
        if not str(result.get('coverage_basis', '')).startswith('complete_model_body_exposure_or_exact_body_label_transfer'):
            errors.append('fanout_coverage_basis_missing')
        final = (result.get('trace') or [{}])[-1]
        if final.get('exact_body_transferred_records') != len(fanout_ids):
            errors.append('fanout_final_record_count_mismatch')
    if rounds:
        actual_cells = {(rid, name) for rid, fields in (result.get('record_facts') or {}).items() for name in fields}
        if set(expected_values) != actual_cells:
            errors.append('raw_response_accepted_cells_not_retained')
        actual_model_ranges = {_span_bounds(value) for value in result.get('model_exposed_spans') or []}
        if actual_model_ranges != body_ranges:
            errors.append('model_exposure_claims_headers_or_omits_body')
        if ('body_exposed_spans' in result
                and {_span_bounds(value) for value in result['body_exposed_spans']} != body_ranges):
            errors.append('body_exposure_summary_mismatch')
        intervals = []
        for start, end in sorted(body_ranges):
            if intervals and start <= intervals[-1][1]:
                intervals[-1][1] = max(intervals[-1][1], end)
            else:
                intervals.append([start, end])
        covered = sum(end - start for start, end in intervals)
        unread, cursor = [], 0
        for start, end in intervals:
            if cursor < start:
                unread.append([cursor, start])
            cursor = end
        if cursor < len(context):
            unread.append([cursor, len(context)])
        coverage = result.get('model_source_coverage') or {}
        if (coverage.get('source_chars') != len(context) or coverage.get('covered_chars') != covered
                or not isinstance(coverage.get('fraction'), (int, float))
                or not math.isclose(coverage.get('fraction', -1), covered / len(context) if context else 1., abs_tol=1e-15)
                or coverage.get('unread_ranges') != unread):
            errors.append('model_body_coverage_mismatch')
    return {'protocol_rounds': len(rounds), 'raw_responses_replayed': response_replays,
            'strict_batch_rejections_verified': strict_batch_rejections, 'projected_body_ranges': len(body_ranges),
            'projection_bindings_checked': len(known_bindings), 'fanout_records_checked': len(fanout_ids),
            'fanout_source_ranges_checked': len(fanout_reference_ranges), 'full_record_span_is_model_exposure_proof': False,
            'violations': sorted(set(errors)), 'passed': not errors}


def audit_conditional_group(result: dict, collection) -> dict:
    """Independent count/bound reduction, without invoking production operators.

    Accepted model classes are premises rather than verified ground truth.
    Source parser/scope membership is shared; the relational reduction is not.
    """
    plan, actual = result.get('plan') or {}, result.get('aggregate') or {}
    scope = collection.scope(plan.get('filters'))
    labels = [str(label).strip().casefold() for label in plan.get('labels') or []]
    predicate = plan.get('semantic_filter') or {}
    raw_values = predicate.get('value')
    wanted = raw_values if isinstance(raw_values, list) else [raw_values]
    wanted = {str(value).strip().casefold() for value in wanted}
    operator, dimension = predicate.get('operator'), plan.get('group_by')
    field, reduction = plan.get('field', 'label'), plan.get('reduction')
    counts, populations, known_groups, unknown_groups = Counter(), Counter(), Counter(), Counter()
    known = unknown = conflicts = missing = 0
    for source_record in scope.records:
        group = source_record.fields.get(dimension)
        valid = isinstance(group, (str, int, float, bool)) and group != '' and not (
            isinstance(group, float) and not math.isfinite(group))
        key = str(group) if valid else None
        if key is None:
            missing += 1
        else:
            counts.setdefault(key, 0)
            populations[key] += 1
        cell = ((result.get('record_facts') or {}).get(source_record.record_id) or {}).get(field)
        candidates = cell.get('candidates') or [] if cell else []
        is_conflict = len(candidates) > 1
        value = candidates[0] if len(candidates) == 1 else source_record.fields.get(field) if cell is None else None
        conflicts += is_conflict
        value_key = str(value).strip().casefold() if value is not None else None
        if is_conflict or value_key is None or (labels and labels.count(value_key) != 1):
            unknown += 1
            if key is not None:
                unknown_groups[key] += 1
            continue
        known += 1
        if key is not None:
            known_groups[key] += 1
            matched = value_key in wanted
            matched = not matched if operator in {'ne', 'not_in'} else matched
            counts[key] += matched
    potential = None if scope.unindexed_count is None else scope.unindexed_count + len(scope.uncertain_records) + missing
    bounds = {key: {'lower': counts[key], 'upper': counts[key] + unknown_groups[key] + potential if potential is not None else None}
              for key in sorted(counts)}
    coverage_complete = bool(scope.complete and not unknown and not missing)
    expected = {'counts': dict(sorted(counts.items())), 'bounds': bounds,
                'group_populations': dict(sorted(populations.items())),
                'group_known_counts': dict(sorted(known_groups.items())),
                'group_unknown_counts': dict(sorted(unknown_groups.items())),
                'known_count': known, 'unknown_count': unknown, 'conflict_count': conflicts,
                'missing_group_count': missing, 'total_count': len(scope.records),
                'scope_complete': scope.complete, 'scope_uncertain_count': len(scope.uncertain_records),
                'unindexed_count': scope.unindexed_count, 'coverage_complete': coverage_complete}
    violations = ['independent_conditional_group_' + key + '_mismatch' for key, value in expected.items() if actual.get(key) != value]
    winners, value, complete = [], None, coverage_complete
    if coverage_complete:
        if reduction == 'counts':
            value = expected['counts']
        elif counts:
            optimum = (max if reduction == 'argmax' else min)(counts.values())
            winners = sorted(key for key in counts if counts[key] == optimum)
            value = winners[0] if len(winners) == 1 else winners
    elif reduction != 'counts' and scope.complete and not missing and bounds:
        # A strict bounded winner must first be the UNIQUE best lower bound
        # (argmax) or upper bound (argmin). Test only that necessary candidate,
        # then compare against every competitor in one separate linear scan.
        # This differs from production's best-two-bound implementation.
        endpoint = 'lower' if reduction == 'argmax' else 'upper'
        optimum = (max if reduction == 'argmax' else min)(entry[endpoint] for entry in bounds.values())
        candidates = [key for key in bounds if bounds[key][endpoint] == optimum]
        if len(candidates) == 1:
            key = candidates[0]
            others = [entry for other, entry in bounds.items() if other != key]
            certified = (all(bounds[key]['lower'] > other['upper'] for other in others) if reduction == 'argmax' else
                         all(bounds[key]['upper'] < other['lower'] for other in others))
            if certified:
                winners = [key]
                complete, value = True, key
    if actual.get('complete') != complete or actual.get('value') != value or actual.get('ties') != (sorted(winners) if complete else []):
        violations.append('independent_conditional_group_winner_certificate_mismatch')
    return {'group_count': len(counts), 'physical_records_checked': len(scope.records),
            'semantic_label_truth_verified': False, 'production_reducer_called': False,
            'source_parser_and_scope_shared': True, 'violations': violations, 'passed': not violations}


def audit_v3_mechanisms(record: dict, sample: dict, budget: dict | None = None) -> dict:
    """Explicit source-text audit; no labels, provider, tokenizer, or model I/O.

    Checks artifact conservation and provenance, not whether a semantic label
    or narrative assertion is actually entailed by its source quotation.
    """
    if record.get('method') == 'adaptive_v31':
        from long_context_agent.legacy_v31.fact_ledger import FactLedger, parse_records
        from long_context_agent.legacy_v31.adaptive_evidence_v3 import _aggregate, _compact_aggregate, _fact_messages
    else:
        from long_context_agent.fact_ledger import FactLedger, parse_records
        from long_context_agent.adaptive_evidence_v3 import _aggregate, _compact_aggregate, _fact_messages
    canonical = lambda value: json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    hashed = lambda value: hashlib.sha256(canonical(value).encode()).hexdigest()
    result, context = record.get('result') or {}, str(sample['context'])
    trace = result.get('trace') or []
    mode, violations = result.get('mode'), []
    source_sha = hashlib.sha256(context.encode()).hexdigest()
    if result.get('source_sha256') != source_sha or record.get('context_sha256') != source_sha:
        violations.append('source_hash_mismatch')
    rounds = [x for x in trace if x.get('action') in {'classify_records', 'fact_review'}]
    stagnant = 0
    already_stopped = False
    events = []
    for index, review in enumerate(rounds):
        current = (review.get('fact_events') or []) + (review.get('fanout_events') or [])
        statuses = {'accepted'} if review['action'] == 'classify_records' else {'accepted', 'conflict', 'corroborated'}
        gain = sum(x.get('status') in statuses for x in current)
        stagnant = 0 if gain else stagnant + 1
        if already_stopped:
            violations.append(f'round_{index}:continued_after_stagnation_limit')
        if review.get('information_gain') != gain or review.get('stagnant_rounds') != stagnant:
            violations.append(f'round_{index}:information_gain_or_stagnation_mismatch')
        if stagnant > 2:
            violations.append(f'round_{index}:zero_gain_run_exceeds_two')
        already_stopped = stagnant >= 2
        events.extend(current)
    checks = {'mode': mode, 'source_sha256': source_sha, 'review_rounds': len(rounds),
              'provenance_references_checked': 0, 'final_prompt_hash_matches_full_retained_ledger': None,
              'semantic_entailment_verified': False}

    def span_text(span):
        match = re.fullmatch(r'source_chars (\d+):(\d+)', str(span))
        if not match:
            return None
        start, end = map(int, match.groups())
        return context[start:end] if 0 <= start < end <= len(context) else None

    if mode == 'adaptive_fact_aggregation_v3':
        collection = parse_records(context)
        by_id = collection.by_id
        protocol_checks = audit_record_protocol(record, context, collection, budget)
        checks['record_protocol_audit'] = protocol_checks
        violations.extend(protocol_checks['violations'])
        ledger = FactLedger(collection)
        facts = result.get('record_facts') or {}
        event_cells = {(e.get('record_id'), e.get('field')) for e in events
                       if e.get('status') in {'accepted', 'duplicate', 'conflict'}}
        retained_cells = {(rid, field) for rid, fields in facts.items() for field in fields}
        if retained_cells != event_cells:
            violations.append('record_event_cells_not_conserved')
        for rid, fields in facts.items():
            source_record = by_id.get(rid)
            if source_record is None:
                violations.append('retained_record_id_not_in_source')
                continue
            for field, cell in fields.items():
                candidates = cell.get('candidates') or []
                if not candidates or cell.get('status') != ('conflict' if len(candidates) > 1 else 'known'):
                    violations.append('record_candidate_status_mismatch')
                if cell.get('value') != (candidates[0] if len(candidates) == 1 else None):
                    violations.append('record_candidate_value_mismatch')
                evidence = cell.get('evidence') or []
                if not evidence:
                    violations.append('record_fact_without_provenance')
                for ref in evidence:
                    checks['provenance_references_checked'] += 1
                    if ref.get('source_sha256') != source_sha or ref.get('source_span') != source_record.source_span or span_text(ref.get('source_span')) != source_record.text:
                        violations.append('record_fact_provenance_mismatch')
                for value in candidates:
                    ledger.add(rid, value, field=field, source_span=source_record.source_span, source_sha256=source_sha)
        plan = result.get('plan') or {}
        if plan.get('operation') == 'conditional_group':
            checks['conditional_group_audit'] = audit_conditional_group(result, collection)
            violations.extend(checks['conditional_group_audit']['violations'])
        try:
            state = ledger.state(plan.get('filters'), field=plan.get('field', 'label'), compact=True)
            aggregate = _compact_aggregate(_aggregate(ledger, plan))
            if canonical(state) != canonical(result.get('fact_state')):
                violations.append('reconstructed_fact_state_or_digest_mismatch')
            if canonical(aggregate) != canonical(result.get('aggregate')):
                violations.append('reconstructed_deterministic_aggregate_mismatch')
            final = [x for x in trace if x.get('action') == 'deterministic_finalize']
            if len(final) != 1 or trace[-1].get('action') != 'deterministic_finalize':
                violations.append('deterministic_finalize_missing_or_not_last')
            elif (final[0].get('fact_digest') != state['fact_digest'] or final[0].get('retained_facts') != len(facts)
                  or canonical(final[0].get('aggregate')) != canonical(aggregate) or final[0].get('model_final_call') is not False):
                violations.append('deterministic_final_fact_or_aggregate_loss')
        except (KeyError, TypeError, ValueError):
            violations.append('record_state_reconstruction_failed')
        attempts = Counter(rid for review in rounds for rid in review.get('selected_record_ids', []))
        if any(value > 2 for value in attempts.values()):
            violations.append('record_classification_attempt_limit')
        checks.update(retained_record_count=len(facts), retained_cell_count=len(retained_cells))
    elif mode == 'adaptive_fact_memory_v3':
        retained = result.get('retained_facts') or []
        facts = {str(x.get('subject', '')).casefold() + '\u0000' + str(x.get('predicate', '')).casefold(): x for x in retained}
        expected_keys, conflicts, corroborations = set(), Counter(), Counter()
        exposed = {span for review in rounds for span in review.get('new_spans', [])}
        for review in rounds:
            for event in review.get('fact_events', []):
                status, key = event.get('status'), event.get('key')
                if status in {'accepted', 'conflict', 'corroborated'}:
                    expected_keys.add(key)
                    conflicts[key] += status == 'conflict'
                    corroborations[key] += status == 'corroborated'
            if review.get('retained_facts') != len(expected_keys):
                violations.append('review_retained_key_count_mismatch')
        if len(facts) != len(retained) or set(facts) != expected_keys:
            violations.append('narrative_accepted_or_conflict_key_lost')
        for key, fact in facts.items():
            candidates = fact.get('candidates') or []
            if len(candidates) != 1 + conflicts[key] or fact.get('status') != ('conflict' if len(candidates) > 1 else 'known'):
                violations.append('narrative_conflict_candidates_lost')
            supports = sum(len(x.get('supporting_sources') or []) for x in candidates)
            if supports != corroborations[key]:
                violations.append('narrative_corroboration_source_lost')
            for candidate in candidates:
                for ref in [candidate, *(candidate.get('supporting_sources') or [])]:
                    checks['provenance_references_checked'] += 1
                    raw, quote = span_text(ref.get('source_span')), str(ref.get('quote') or '')
                    if not quote or raw is None or ref.get('source_span') not in exposed or re.sub(r'\s+', ' ', quote) not in re.sub(r'\s+', ' ', raw):
                        violations.append('narrative_provenance_mismatch')
        final = trace[-1] if trace else {}
        if final.get('action') not in {'final_answer', 'final_answer_unavailable'} or final.get('retained_fact_count') != len(facts):
            violations.append('narrative_final_retained_count_mismatch')
        if final.get('action') == 'final_answer':
            if final.get('fact_digest') != hashed(facts):
                violations.append('narrative_final_digest_mismatch')
            if budget is not None:
                # Reconstruct the actual final wire hash with all retained facts.
                # Only acquisition-gap wording may be shortened by production.
                wire = record.get('ledger') or []
                if not wire:
                    violations.append('narrative_final_request_missing')
                else:
                    prior = wire[:-1]
                    notice = (f'Experiment resource limits: input plus output per call <= {budget["context_tokens"]} tokens; '
                              f'remaining physical generation requests INCLUDING this one: {max(0, budget["max_calls"] - len(prior))}; '
                              f'remaining cumulative input plus output tokens: {max(0, budget["total_tokens"] - sum(x["charged_tokens"] for x in prior))}; '
                              f'maximum output per request: {budget["max_output_tokens"]} tokens. '
                              'Root, recursive and summary requests all share these limits. '
                              'Complete the requested answer within these resources.')
                    gap_versions = [result.get('unresolved_gaps') or []]
                    if rounds:
                        gap_versions.extend([rounds[-1].get('gaps_before') or [], rounds[-1].get('gaps_after') or []])
                    gap_versions += [[{'id': g.get('id', ''), 'kind': g.get('kind', 'missing'), 'query': str(g.get('query', ''))[:120]} for g in gaps[:8]] for gaps in list(gap_versions)]
                    hashes = []
                    for gaps in gap_versions:
                        messages = _fact_messages(str(sample.get('question', '')), facts, gaps, final=True)
                        messages[0] = {'role': 'system', 'content': notice + '\n\n' + messages[0]['content']}
                        hashes.append(hashlib.sha256(json.dumps(messages, sort_keys=True, ensure_ascii=False).encode()).hexdigest())
                    checks['final_prompt_hash_matches_full_retained_ledger'] = wire[-1].get('prompt_sha256') in hashes
                    if not checks['final_prompt_hash_matches_full_retained_ledger']:
                        violations.append('final_wire_prompt_does_not_contain_complete_retained_ledger')
        checks.update(retained_fact_count=len(facts), retained_conflicts=sum(x.get('status') == 'conflict' for x in facts.values()))
    elif mode != 'direct_pass_v3':
        violations.append('unrecognized_v3_mechanism_mode')
    checks['violations'] = sorted(set(violations))
    checks['passed'] = not violations
    return checks


def verify_official_scores(output: Path, frozen: dict, manifest: dict, records: dict, report: dict) -> dict:
    """Only called after the entire frozen cross-product passes execution checks."""
    if not report['complete'] or report['violations']:
        raise ValueError('Cannot open labels before all predictions pass execution audit')
    manifest_path = Path(frozen['data_manifest'])
    scorer_path = manifest_path.parent / 'source/eval_helpers_original.py'
    labels_path = manifest_path.parent / 'labels.json'
    if manifest.get('scorer_commit') != OFFICIAL_COMMIT or digest_file(scorer_path) != frozen['official_scorer_sha256']:
        raise ValueError('Official scorer identity differs from the pinned freeze')
    # The first label byte is read only below this completion gate.
    labels_raw = labels_path.read_bytes()
    if hashlib.sha256(labels_raw).hexdigest() != frozen['labels_sha256']:
        raise ValueError('Label hash differs from the pre-inference sealed identity')
    labels = json.loads(labels_raw)
    if set(labels) != set(frozen['sample_ids']):
        raise ValueError('Label identities do not match frozen sample identities')
    official = load_official_scorer(scorer_path)
    existing_path = output / 'audit.json'
    existing = read_json(existing_path) if existing_path.exists() else None
    existing_rows = {(x['id'], x['method']): x for x in existing['samples']} if existing else {}
    if existing and set(existing_rows) != set(records):
        raise ValueError('Published audit does not contain exactly the frozen pairs')
    rows, mismatches = [], []
    for pair, record in records.items():
        value = official(labels[pair[0]], str(record.get('prediction') or ''), frozen['model'])
        score = float(value['score'])
        row = {'id': pair[0], 'method': pair[1], 'official_score': score, 'full_credit': score == 1.0}
        rows.append(row)
        if existing:
            published = existing_rows[pair]
            if not math.isclose(float(published['official_score']), score, abs_tol=1e-15, rel_tol=1e-12) or published['full_credit'] != row['full_credit']:
                mismatches.append({'id': pair[0], 'method': pair[1], 'kind': 'per_pair_score'})
    summary = {}
    for method in frozen['methods']:
        group = [x for x in rows if x['method'] == method]
        summary[method] = {'n': len(group), 'mean_official_score': statistics.mean(x['official_score'] for x in group),
                           'full_credit': sum(x['full_credit'] for x in group)}
        if existing:
            published = existing['summary'][method]
            if not math.isclose(published['mean_official_score'], summary[method]['mean_official_score'], abs_tol=1e-15, rel_tol=1e-12) or published['full_credit'] != summary[method]['full_credit']:
                mismatches.append({'method': method, 'kind': 'method_summary'})
    access = {'time_unix': time.time(), 'all_predictions_present_before_label_read': True,
              'expected_pairs': report['expected_pairs'], 'prediction_files': report['prediction_files'],
              'labels_sha256': frozen['labels_sha256'], 'scorer_sha256': digest_file(scorer_path),
              'scorer_commit': OFFICIAL_COMMIT, 'extracted_functions': ['synth_attempt_answer_parse', 'synth_process_response'],
              'auditor_sha256': digest_file(Path(__file__)), 'no_model_requests': True}
    atomic_json(output / 'independent_scoring_access.json', access)
    return {'verified_against_published_audit': existing is not None, 'mismatches': mismatches,
            'summary': summary, 'pair_scores': rows, 'access': access}


def audit_execution(output: Path, *, root: Path = ROOT, allow_incomplete=False, verify_scores=False, verify_mechanisms=False) -> dict:
    started = time.time()
    frozen = read_json(output / 'freeze.json')
    freeze_hash = digest_file(output / 'freeze.json')
    manifest_path = Path(frozen['data_manifest'])
    manifest = read_json(manifest_path)
    violations, warnings, pairs = [], [], []
    source = source_checks(output, frozen, root)
    if any(source[k] for k in ('missing_snapshot_files', 'snapshot_hash_mismatches', 'current_declared_file_differences')):
        violations.append('source_or_snapshot_hash_mismatch')
    if digest_file(manifest_path) != frozen['data_manifest_sha256']:
        violations.append('data_manifest_hash_mismatch')
    # Hashing is opaque: no inference question/context text is parsed by audit.
    if digest_file(frozen['inputs']) != frozen['inputs_sha256'] or frozen['inputs_sha256'] != manifest['inputs_sha256']:
        violations.append('inputs_hash_mismatch')
    if frozen['labels_sha256'] != manifest['labels_sha256'] or frozen['official_scorer_sha256'] != manifest['scorer_sha256']:
        violations.append('sealed_label_or_scorer_identity_mismatch')
    if len(set(frozen['sample_ids'])) != len(frozen['sample_ids']) or len(set(frozen['methods'])) != len(frozen['methods']):
        violations.append('duplicate_frozen_identity')
    selected = {x['id']: x for x in manifest['selected']}
    if set(selected) != set(frozen['sample_ids']):
        violations.append('manifest_sample_identity_mismatch')
    guard = guard_checks(output, frozen, manifest_path)
    for key in ('profile_matches_freeze', 'labels_literal_denied', 'probe_literal_denied', 'probe_read_denied', 'guard_verified', 'labels_not_opened_by_runner', 'verification_profile_matches'):
        if not guard[key]:
            violations.append('label_guard:' + key)
    observed_runtime = runtime_identity()
    if observed_runtime != frozen['runtime']:
        violations.append('runtime_identity_changed')
    expected = {(sample, method) for sample in frozen['sample_ids'] for method in frozen['methods']}
    records, missing = {}, []
    latest_mtime = 0
    for sample, method in sorted(expected):
        name = f'{sample}__{method}.json'
        prediction, journal = output / 'predictions' / name, output / 'journals' / name
        if not prediction.exists():
            missing.append(name)
            continue
        latest_mtime = max(latest_mtime, prediction.stat().st_mtime)
        record = read_json(prediction)
        records[sample, method] = record
        prefix = f'{sample}__{method}:'
        if record.get('id') != sample or record.get('method') != method or record.get('freeze_sha256') != freeze_hash:
            violations.append(prefix + 'prediction_identity_or_freeze')
        if record.get('context_sha256') != selected.get(sample, {}).get('context_sha256'):
            violations.append(prefix + 'source_hash_mismatch')
        ledger = record.get('ledger', [])
        if not journal.exists():
            violations.append(prefix + 'journal_missing')
        else:
            j = read_json(journal)
            if j.get('id') != sample or j.get('method') != method or j.get('freeze_sha256') != freeze_hash or j.get('ledger') != ledger:
                violations.append(prefix + 'journal_identity_or_ledger_mismatch')
        accounting = audit_ledger(ledger, frozen['budget'])
        violations.extend(prefix + value for value in accounting['violations'])
        warnings.extend(prefix + value for value in accounting['warnings'])
        summary = record.get('usage') or {}
        for key in ('physical_requests', 'prompt_tokens', 'completion_tokens', 'charged_tokens'):
            if summary.get(key) != accounting[key]:
                violations.append(prefix + 'summary_mismatch:' + key)
        if summary.get('budget') != frozen['budget'] or summary.get('within_budget') is not True:
            violations.append(prefix + 'summary_budget_mismatch')
        if summary.get('all_usage_known') != (accounting['unknown_usage_requests'] == 0):
            violations.append(prefix + 'summary_usage_known_mismatch')
        pairs.append({'id': sample, 'method': method, 'prediction_file_sha256': digest_file(prediction),
                      **{k: v for k, v in accounting.items() if k not in {'violations', 'warnings'}}})
    expected_names = {f'{s}__{m}.json' for s, m in expected}
    pending_journals, extra_journals = [], []
    for path in sorted((output / 'journals').glob('*.json')):
        if path.name not in expected_names:
            extra_journals.append(path.name)
            continue
        journal = read_json(path)
        pair = journal.get('id'), journal.get('method')
        if pair in records:
            continue
        prefix = path.name + ':'
        if pair not in expected or path.name != f'{pair[0]}__{pair[1]}.json' or journal.get('freeze_sha256') != freeze_hash:
            violations.append(prefix + 'pending_journal_identity_or_freeze')
        accounting = audit_ledger(journal.get('ledger', []), frozen['budget'])
        violations.extend(prefix + x for x in accounting['violations'])
        warnings.extend(prefix + x for x in accounting['warnings'])
        pending_journals.append({'id': pair[0], 'method': pair[1],
                                 **{k: v for k, v in accounting.items() if k not in {'violations', 'warnings'}}})
    extra_predictions = sorted(x.name for x in (output / 'predictions').glob('*.json')
                               if x.name not in expected_names)
    if extra_predictions:
        violations.append('unexpected_prediction_files')
    if extra_journals:
        violations.append('unexpected_journal_files')
    if missing and not allow_incomplete:
        violations.append('incomplete_prediction_cross_product')
    scoring_access = read_json(output / 'scoring_access.json') if (output / 'scoring_access.json').exists() else None
    if scoring_access and (missing or scoring_access.get('labels_first_scored_at_unix', 0) < latest_mtime):
        violations.append('labels_scored_before_predictions_complete')
    report = {'audit_version': 3, 'started_at_unix': started, 'finished_at_unix': time.time(),
              'freeze_sha256': freeze_hash, 'model': frozen['model'], 'budget': frozen['budget'],
              'expected_pairs': len(expected), 'prediction_files': len(records), 'complete': not missing,
              'missing_predictions': missing, 'unexpected_predictions': extra_predictions,
              'unexpected_journals': extra_journals, 'pending_journal_accounting': pending_journals,
              'pairs_by_method': dict(Counter(x['method'] for x in pairs)),
              'source_and_snapshot_checks': source, 'runtime_identity_matches': observed_runtime == frozen['runtime'],
              'guard_checks': guard, 'last_prediction_mtime_unix': latest_mtime,
              'physical_generation_requests': sum(x['physical_requests'] for x in pairs + pending_journals),
              'charged_tokens': sum(x['charged_tokens'] for x in pairs + pending_journals),
              'unknown_usage_requests': sum(x['unknown_usage_requests'] for x in pairs + pending_journals),
              'pair_accounting': pairs, 'violations': violations, 'warnings': warnings,
              'labels_opened_by_this_audit': False, 'labels_hashed_by_this_audit': False,
              'inference_input_semantics_parsed': False, 'no_model_requests': True}
    if verify_mechanisms:
        samples = {row['id']: row for row in (json.loads(line) for line in Path(frozen['inputs']).read_text().splitlines() if line.strip())}
        mechanisms = []
        for (sample, method), record in records.items():
            if method not in {'adaptive_v3', 'adaptive_v31'}:
                continue
            checks = audit_v3_mechanisms(record, samples[sample], frozen['budget'])
            mechanisms.append({'id': sample, **checks})
            violations.extend(f'{sample}:mechanism:' + x for x in checks['violations'])
        report['v3_mechanism_checks'] = mechanisms
        report['inference_input_semantics_parsed'] = True
    if verify_scores:
        report['official_scorer_verification'] = verify_official_scores(output, frozen, manifest, records, report)
        report['labels_opened_by_this_audit'] = report['labels_hashed_by_this_audit'] = True
        if report['official_scorer_verification']['mismatches']:
            report['violations'].append('independent_official_scorer_mismatch')
    report['passed'] = not report['violations']
    report['finished_at_unix'] = time.time()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--allow-incomplete', action='store_true', help='Audit existing records without treating missing pairs as a violation; never relaxes scoring gate')
    parser.add_argument('--verify-scores', action='store_true', help='Explicitly authorize post-completion label access and independent official AST scoring')
    parser.add_argument('--verify-mechanisms', action='store_true', help='Explicitly parse inference source text to verify V3 provenance/conservation; never opens labels')
    args = parser.parse_args()
    report = audit_execution(args.output, allow_incomplete=args.allow_incomplete, verify_scores=args.verify_scores,
                             verify_mechanisms=args.verify_mechanisms)
    destination = args.report or args.output / 'execution_integrity_v3.json'
    atomic_json(destination, report)
    print(json.dumps({k: report[k] for k in ('complete', 'expected_pairs', 'prediction_files', 'passed', 'violations', 'physical_generation_requests', 'charged_tokens', 'labels_opened_by_this_audit')}, indent=2))
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
