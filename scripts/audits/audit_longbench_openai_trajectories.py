"""Offline trajectory audit, only after original 75 + recovery 39 are scored.

No HTTP, provider tokenizer, labels, historical answers, or dataset cache is
opened. The production source selectors and controller are re-executed from
label-free inputs; exact recorded count events and stored model responses are
replayed once. Independent checks bind every rendered evidence window to its
original source and verify request hashes, reservations, and token ledgers.

Only this unfrozen work helper is new. It has not been run on live predictions
during inference. Invoke after the parent confirms all 114 terminal and scored:

  python work/longbench_openai_limits_20261102/audit_trajectories.py --after-all-114

``--self-test`` uses an in-memory scripted fixture, never experiment files.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from long_context_agent.experiment_budget import BudgetExhausted, BudgetIntegrityError, ExperimentBudget
from long_context_agent.openai_longbench_bridge import (
    LongBenchEvidenceConfig, LongBenchResponsesTransport, run_longbench_harness,
)
from long_context_agent.openai_responses_transport import (
    OpenAIResponseNotCompletedError, OpenAIResponsesBudgetedClient, OpenAIResponsesTransportError,
)
from scripts.audit_context_holdout_execution_v3 import audit_ledger
from scripts.benchmark_longbench_v2_full import chunk_items
from scripts.benchmark_openai_transfer import CostCappedClient
from scripts.longbench_v2_protocol import (
    LongBenchRow, evidence_window_groups, extract_choice, official_prompt, verify_evidence_candidates,
)


EXPECTED_FREEZE = '941c86120e66020439571fbe8888999c643ebcf83fc315d7ec7e1e0e1045208d'
EXPECTED_RECOVERY_FREEZE = 'febefea46171ba7d843104aa8857bff82670b597e6b0bb0de6b640e7c7f312c1'
EXPECTED_CONTROL_FREEZE = 'dfb25fdb5cd26067a881dca5747b78ded10c7a8e5389091eacb2529a26890c3f'
MODELS = ('gpt-6-sol', 'gpt-6-luna', 'gpt-6.1-terra')
NATIVE = 1_050_000
HEADER = re.compile(r'\[query_mode [^\n]*?\| source_chars (\d+):(\d+) \|[^\n]*\]\n')
STRICT_PATTERNS = (r'The correct answer is \(([A-D])\)', r'The correct answer is ([A-D])')
OFFICIAL_PARSER_SOURCE_SHA = 'ab63f77866a1c0dc770582bc3fe6b014c3b4be4667399b0ee267075780c6a138'
OFFICIAL_PARSER_FUNCTION_SHA = 'e602098fcae0fb006bd6f41f7a3caf0d317999d40c9fbc9c973a034f03f7ba54'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text_hash(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def object_hash(value):
    # This intentionally matches the frozen transport and BudgetedClient
    # JSON serialization, including the default separators.
    return text_hash(json.dumps(value, sort_keys=True, ensure_ascii=False))


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def require(condition, code):
    if not condition:
        raise ReplayViolation(code)


class ReplayViolation(RuntimeError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


class ReplayUnavailable(RuntimeError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


class _NoopMonetaryGate:
    """Keep original can_call/count ordering; money is audited separately."""

    def reserve(self, **kwargs):
        return {}

    def settle(self, reservation, entry):
        pass

    def save(self):
        pass


class _CallRecordingCostGate(CostCappedClient):
    """Retain private candidate-object provenance even for admission failures."""

    def __init__(self, client, sample_id):
        super().__init__(client, _NoopMonetaryGate(), sample_id)
        self.call_events = []

    def complete(self, messages, max_tokens=2048):
        user = next((str(message.get('content') or '') for message in reversed(messages)
                     if message.get('role') == 'user'), '')
        kind = ('candidate' if user.startswith('You are answering a multiple-choice question after an evidence-aware')
                else 'verifier' if user.startswith('You are a strict evidence verifier for a multiple-choice question.')
                else 'direct' if user.startswith('Please read the following text and answer the question below.')
                else 'other')
        before = len(self.client.ledger)
        event = {'kind': kind, 'returned': False, 'raw_reply': None, 'physical_request': None}
        self.call_events.append(event)
        try:
            reply = super().complete(messages, max_tokens=max_tokens)
            event.update(returned=True, raw_reply=str(reply or ''))
            return reply
        finally:
            if len(self.client.ledger) > before:
                event['physical_request'] = self.client.ledger[-1]['physical_request']


def official_strict_extract(response):
    """Independent equivalent of the pinned official extract_answer function."""
    if not isinstance(response, str):
        return None
    response = response.replace('*', '')
    for pattern in STRICT_PATTERNS:
        match = re.search(pattern, response)
        if match:
            return match.group(1)
    return None


def official_parser_provenance():
    directory = ROOT/'work/longbench_openai_limits_20261102'
    metadata = load(directory/'official_parser_provenance.json')
    source_path = directory/'official_pred_source.py'
    require(digest(source_path) == metadata.get('source_sha256') == OFFICIAL_PARSER_SOURCE_SHA,
            'official_parser_snapshot_source_hash_mismatch')
    source = source_path.read_text()
    node = next((node for node in ast.parse(source).body
                 if isinstance(node, ast.FunctionDef) and node.name == 'extract_answer'), None)
    require(node is not None, 'official_extract_answer_function_missing')
    segment = ast.get_source_segment(source, node)
    require(text_hash(segment) == metadata.get('function_sha256') == OFFICIAL_PARSER_FUNCTION_SHA,
            'official_parser_function_hash_mismatch')
    searches = [call for call in ast.walk(node) if isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name)
                and call.func.value.id == 're' and call.func.attr == 'search']
    require(len(searches) == 2 and all(len(call.args) == 2 and not call.keywords for call in searches)
            and tuple(call.args[0].value for call in searches) == STRICT_PATTERNS,
            'official_parser_patterns_or_case_policy_changed')
    require("response.replace('*', '')" in segment, 'official_asterisk_removal_policy_changed')
    # The upstream module is data only: its imports, client setup and dataset
    # loading are never executed. Only our equivalent two-search function runs.
    return {key: metadata[key] for key in ('upstream_requested_url', 'upstream_commit', 'pinned_raw_url',
        'source_sha256', 'function_sha256', 'function_name', 'asterisks_removed', 'case_sensitive', 'first_match')}


def selected_raw_reply(record, row, groups, call_events):
    """Follow the frozen LOCAL selection; strict extraction cannot reselect."""
    if record['arm'] != 'harness':
        event = next((event for event in reversed(call_events) if event['kind'] == 'direct'), None)
        if event and event['returned']:
            return event['raw_reply'], {'kind': ('actual_headtail_raw_reply' if record['arm'] == 'headtail_control'
                                                 else 'actual_direct_raw_reply'),
                'physical_request': event['physical_request'], 'candidate_index': None}
        return None, {'kind': 'no_accepted_direct_raw_reply', 'physical_request': None, 'candidate_index': None}
    result = record.get('result') or {}
    if not result or not groups:
        return None, {'kind': 'no_returned_harness_artifact', 'physical_request': None, 'candidate_index': None}
    candidates = result.get('candidate_results') or []
    evidence = [str(group.get('text') or '') for group in groups]
    if len(candidates) == len(evidence)+1 and evidence:
        evidence.append(evidence[min(2, len(evidence)-1)])
    require(len(candidates) == len(evidence), 'candidate_evidence_mapping_not_reconstructable')
    verification = verify_evidence_candidates(row,
        [{**candidate, 'evidence_text': evidence[index]} for index, candidate in enumerate(candidates)])
    winner, index = verification.get('verified_winner'), verification.get('selected_candidate_index')
    option = result.get('option_verification') or {}
    if option.get('strong_override'):
        matching = next(((position, candidate) for position, candidate in enumerate(candidates)
                         if candidate.get('vote_choice') == option.get('winner')), None)
        if matching is not None:
            index, winner = matching[0], option['winner']
    require(winner == result.get('selected_winner'), 'fixed_controller_winner_provenance_mismatch')
    if index is None:
        return None, {'kind': 'controller_has_no_selected_candidate', 'physical_request': None, 'candidate_index': None}
    require(type(index) is int and 0 <= index < len(candidates), 'selected_candidate_index_invalid')
    events = [event for event in call_events if event['kind'] == 'candidate']
    require(index < len(events), 'selected_candidate_call_event_missing')
    event = events[index]
    if not event['returned']:
        return None, {'kind': 'selected_candidate_has_no_accepted_reply',
            'physical_request': event['physical_request'], 'candidate_index': index}
    return event['raw_reply'], {'kind': 'fixed_actual_selected_candidate_raw_reply',
        'physical_request': event['physical_request'], 'candidate_index': index,
        'selection_policy': 'original_local_vote_verification_and_option_override_unchanged',
        'strict_parser_used_to_select_candidate': False}


def strict_choice_receipt(record, row, groups, call_events):
    raw, provenance = selected_raw_reply(record, row, groups, call_events)
    available = raw is not None
    strict, local = official_strict_extract(raw), extract_choice(raw) if available else None
    return {'index': record['index'], 'id': record.get('id', row.item_id), 'model': record['model'], 'arm': record['arm'],
        'record_status': record['status'], 'actual_record_freeze_sha256': record.get('freeze_sha256'),
        'source_context_sha256': record.get('context_sha256'),
        'original_context_sha256': record.get('original_context_sha256', record.get('context_sha256')),
        'raw_reply_available': available, 'raw_reply_sha256': text_hash(raw) if available else None,
        'strict_choice': strict, 'local_raw_reply_choice': local, 'actual_controller_choice': record.get('prediction'),
        'selected_provenance': provenance,
        'official_and_local_raw_parser_agree': strict == local if available else None,
        'official_and_recorded_primary_agree': strict == record.get('prediction') if available else None,
        'local_raw_parser_and_recorded_primary_agree': local == record.get('prediction') if available else None}


def headtail_projection(context, retained):
    """Independent character projection, with no join delimiter or padding."""
    require(type(retained) is int and 2 <= retained <= len(context), 'invalid_headtail_retained_character_count')
    if retained == len(context):
        return context, [[0, len(context)]], None
    head, tail = (retained+1)//2, retained//2
    spans = [[0, head], [len(context)-tail, len(context)]]
    return context[:head]+context[len(context)-tail:], spans, [head, len(context)-tail]


def control_metadata_gate(control, parent, child):
    control = Path(control)
    frozen = load(control/'freeze.json')
    require(digest(control/'freeze.json') == EXPECTED_CONTROL_FREEZE, 'unexpected_headtail_control_freeze')
    count = frozen.get('expected_control_records')
    require(count in (0, 3, 6), 'invalid_registered_control_count')
    run, access = load(control/'run_summary.json'), load(control/'scoring_access.json')
    require(run.get('phase') == 'headtail_control_complete'
            and run.get('planned_control_records') == count and run.get('terminal_control_records') == count,
            'all_registered_headtail_controls_not_terminal')
    require(access.get('all_114_and_control_terminal_before_labels') is True
            and access.get('control_terminal_count') == count and (control/'audit.json').is_file(),
            'headtail_final_scoring_gate_not_satisfied')
    require(frozen.get('parent_freeze_sha256') == EXPECTED_FREEZE
            and frozen.get('recovery_freeze_sha256') == EXPECTED_RECOVERY_FREEZE,
            'headtail_parent_or_recovery_freeze_mismatch')
    require(frozen.get('all_114_terminal_before_control_registration') is True
            and frozen.get('selection_uses_answers_or_model_outputs') is False,
            'headtail_pre_gold_registration_declaration_missing')
    remaining = Decimal(frozen['remaining_before_control_usd'])
    indices, cap = (([143, 260], '13') if remaining >= 13 else
                    ([143], '6.5') if remaining >= Decimal('6.5') else ([], '0'))
    require(frozen.get('planned_indices') == indices and frozen.get('control_cap_usd') == cap
            and count == 3*len(indices) and frozen.get('models') == list(MODELS),
            'headtail_money_only_selection_rule_mismatch')
    expected_budget = {'context_tokens': 800_000, 'total_tokens': 800_000,
                       'max_calls': 1, 'max_output_tokens': 1024, 'token_safety_margin': 16}
    require(frozen.get('budget') == expected_budget, 'headtail_registered_budget_mismatch')
    require(len(frozen['code_sha256']) == 58, 'headtail_frozen_source_count_mismatch')
    require(all(frozen['code_sha256'].get(name) == value for name, value in parent['code_sha256'].items()),
            'headtail_changed_original_frozen_algorithm')
    for name, value in frozen['code_sha256'].items():
        path = Path(name)
        require(not path.is_absolute() and '..' not in path.parts, 'invalid_headtail_source_path')
        require(digest(ROOT/path) == value and digest(control/'frozen_source'/path) == value,
                'headtail_live_or_snapshot_source_hash_mismatch')
    require(frozen['code_sha256'].get('scripts/benchmark_longbench_openai_recovery.py') == child['recovery']['wrapper_sha256'],
            'headtail_recovery_wrapper_mismatch')
    for name, key in (('inputs.jsonl', 'prepared_inputs_sha256'), ('manifest.json', 'manifest_sha256'),
                      ('preparation_count_audit.json', 'preparation_count_audit_sha256'),
                      ('labels_guard.sb', 'guard_profile_sha256')):
        require(digest(control/name) == frozen[key], 'headtail_prepared_artifact_hash_mismatch')
    require(frozen.get('upstream_pred_commit') == 'c5ea10bcd06285223c58dfed76bbc92d22273709',
            'headtail_upstream_parser_reference_mismatch')
    return frozen, access


def audit_headtail_preparation(control, frozen, originals, original_manifest):
    """Recompute every shared character trim using only archived exact counts."""
    manifest = load(control/'manifest.json')
    require(manifest.get('not_bit_identical_to_upstream_token_half') is True
            and manifest.get('generation_requested_during_preparation') is False,
            'headtail_projection_or_count_only_scope_not_disclosed')
    values = [json.loads(line) for line in (control/'inputs.jsonl').read_text().splitlines() if line.strip()]
    by_index = {value['index']: value for value in values}
    metadata = {item['index']: item for item in manifest['selected']}
    require(set(by_index) == set(metadata) == set(frozen['planned_indices'])
            and len(values) == len(by_index), 'headtail_prepared_source_matrix_mismatch')
    counts = load(control/'preparation_count_audit.json')
    transports, clients = {}, {}
    for model in MODELS if frozen['planned_indices'] else ():
        data = counts.get(model) or {}
        transport = ReplayTransport({'model': model, 'ledger': [], 'response_audit': [],
            'count_audit': data.get('count_audit'), 'count_attempt_audit': data.get('physical_count_attempts')}, phase='recovery')
        transports[model] = transport
        clients[model] = OpenAIResponsesBudgetedClient(transport, ExperimentBudget(**frozen['budget']))
    preflight = {item['index']: item for item in original_manifest['selected']}
    projections = []
    for index in frozen['planned_indices']:
        original, trimmed, item = originals[index], by_index[index], metadata[index]
        require(not {'answer', 'gold', 'labels', 'correct'} & trimmed.keys(), 'headtail_inputs_contain_labels')
        require(all(trimmed.get(key) == value for key, value in original.items() if key != 'context'),
                'headtail_changed_question_choices_or_metadata')
        context = original['context']
        estimate = int(preflight[index]['preflight_native_direct']['direct_admission_input_tokens'])
        retained = min(len(context), max(2, int(len(context)*790_000/estimate)))
        history = item.get('preparation_rounds') or []
        require(1 <= len(history) <= 8, 'headtail_preparation_round_count_invalid')
        for number, step in enumerate(history, 1):
            projected, spans, dropped = headtail_projection(context, retained)
            candidate = {**original, 'context': projected}
            prompt = official_prompt(LongBenchRow.from_mapping(candidate, offset=index))
            actual = {model: clients[model].count_messages([{'role': 'user', 'content': prompt}]) for model in MODELS}
            fits = all(count+1024 <= frozen['budget']['context_tokens'] for count in actual.values())
            require(step.get('round') == number and step.get('retained_chars') == len(projected)
                    and step.get('direct_admission_tokens_by_model') == actual and step.get('all_models_fit') is fits,
                    'headtail_preparation_round_or_exact_count_mismatch')
            if fits:
                require(number == len(history), 'headtail_preparation_continued_after_admission_fit')
            else:
                require(number < len(history), 'headtail_final_preparation_round_not_admissible')
                retained = max(2, min(retained-1, int(retained*(790_000/max(actual.values()))*.99)))
        require(trimmed['context'] == projected and item.get('source_spans') == spans
                and item.get('dropped_middle_span') == dropped, 'headtail_exact_character_projection_mismatch')
        require(item.get('id') == original['_id'] and item.get('original_context_chars') == len(context)
                and item.get('retained_chars') == len(projected)
                and item.get('original_context_sha256') == text_hash(context)
                and item.get('trimmed_context_sha256') == text_hash(projected)
                and item.get('trimmed_prompt_sha256') == text_hash(prompt), 'headtail_source_or_prompt_hash_mismatch')
        require(item.get('same_trimmed_source_for_all_models') is True and item.get('inserted_characters') == 0
                and item.get('full_source_retained') is (dropped is None), 'headtail_projection_disclosure_mismatch')
        require(item.get('direct_admission_tokens_by_model') == actual, 'headtail_final_shared_admission_mismatch')
        projections.append({'index': index, 'id': original['_id'],
            'original_context_sha256': text_hash(context), 'trimmed_context_sha256': text_hash(projected),
            'original_context_chars': len(context), 'retained_chars': len(projected),
            'source_spans': spans, 'dropped_middle_span': dropped,
            'inserted_characters': 0, 'question_and_choices_preserved': True,
            'exact_projection_verified': True, 'shared_model_admission_verified': True,
            'preparation_rounds_replayed': len(history)})
    for transport in transports.values():
        transport.assert_consumed()
    return {'projections': projections, 'prepared_source_count': len(projections),
        'preparation_count_events_replayed': sum(t.count_cursor for t in transports.values()),
        'preparation_physical_count_attempts_verified': sum(t.attempt_cursor for t in transports.values()),
        'preparation_retry_operations_verified': sum(t.retried_count_operations for t in transports.values()),
        'not_bit_identical_to_upstream_token_half': True}, by_index, metadata


def replay_headtail(record, value, metadata, private_choices):
    if (record.get('error') or {}).get('type') in {'InterruptedTask', 'ModelBlocked', 'CostCapExceeded'}:
        raise ReplayUnavailable('headtail_control_error_has_no_complete_replayable_trajectory')
    row = LongBenchRow.from_mapping(value, offset=value['index'])
    require(row.answer == '', 'headtail_label_reached_replay')
    transport = ReplayTransport(record, phase='recovery')
    client = OpenAIResponsesBudgetedClient(transport, ExperimentBudget(**record['budget']))
    capped = _CallRecordingCostGate(client, f"headtail_control:{record['index']}")
    prompt = official_prompt(row)
    admission = client.count_messages([{'role': 'user', 'content': prompt}])
    measurements = record.get('measurements') or {}
    require(admission == measurements.get('full_prompt_admission_input_tokens')
            == measurements.get('prepared_admission_input_tokens')
            == metadata['direct_admission_tokens_by_model'][record['model']],
            'headtail_runtime_or_prepared_whole_prompt_count_mismatch')
    require(admission+1024 <= 800_000 and measurements.get('whole_prompt_fits_control_budget') is True,
            'headtail_whole_prompt_exceeded_control_window')
    content = ''
    try:
        content = capped.complete([{'role': 'user', 'content': prompt}], max_tokens=1024)
    except (ReplayUnavailable, ReplayViolation):
        raise
    except Exception as exc:
        require(type(exc).__name__ == (record.get('error') or {}).get('type'), 'headtail_terminal_error_type_mismatch')
    require(extract_choice(content) == record.get('prediction')
            and official_strict_extract(content) == record.get('strict_official_prediction'),
            'headtail_recorded_parser_outcome_mismatch')
    transport.assert_consumed()
    compare_ledgers(client.ledger, record['ledger'])
    receipt = strict_choice_receipt(record, row, None, capped.call_events)
    private_choices.append(receipt)
    return {'status': 'passed', 'wire_prompt_hashes_checked': transport.wire_prompt_hashes_checked,
        'responses_wire_request_hashes_checked': transport.wire_request_hashes_checked,
        'count_events_replayed': transport.count_cursor, 'physical_count_attempts_verified': transport.attempt_cursor,
        'generation_entries_replayed': transport.generation_cursor,
        'known_responses_replayed': transport.known_responses,
        'unknown_delivery_errors_replayed': transport.unknown_errors,
        'recovery_count_operations_retried': transport.retried_count_operations,
        'ledger_matches_stored': True, 'prediction_matches_stored': True,
        'official_raw_parser': {'raw_reply_available': receipt['raw_reply_available'],
            'raw_reply_sha256': receipt['raw_reply_sha256'],
            'official_choice_available': receipt['strict_choice'] is not None,
            'official_and_recorded_primary_agree': receipt['official_and_recorded_primary_agree'],
            'selected_provenance': receipt['selected_provenance']}}


class ReplayTransport(LongBenchResponsesTransport):
    """Replay archived count events and generation entries without I/O."""

    def __init__(self, record, *, phase='original'):
        super().__init__(api_key='OFFLINE_AUDIT_NO_CREDENTIAL', model=record['model'],
                         reasoning_effort='none')
        self.expected_counts = record.get('count_audit') or []
        self.expected_ledger = record.get('ledger') or []
        self.expected_responses = record.get('response_audit')
        self.phase = phase
        self.expected_count_attempts = record.get('count_attempt_audit') if phase == 'recovery' else None
        self.attempt_cursor = 0
        self.retried_count_operations = 0
        self.count_cursor = self.generation_cursor = 0
        self.count_map = {}
        self.count_error_hashes = set()
        self.kind_counts = Counter()
        self.fatal = None
        self.known_responses = self.unknown_errors = 0
        self.wire_prompt_hashes_checked = self.wire_request_hashes_checked = 0
        for entry in self.expected_counts:
            key = entry.get('input_sha256')
            require(isinstance(key, str) and re.fullmatch('[0-9a-f]{64}', key), 'invalid_recorded_count_hash')
            if entry.get('status') == 'ok':
                value = entry.get('input_tokens')
                require(type(value) is int and value >= 0, 'invalid_recorded_exact_count')
                require(key not in self.count_map or self.count_map[key] == value, 'inconsistent_recorded_count_for_hash')
                self.count_map[key] = value
            else:
                self.count_error_hashes.add(key)

    def _post(self, path, body):
        raise ReplayViolation('network_or_provider_tokenizer_call_forbidden')

    def count(self, *, messages=None, text=None, tools=None, tool_choice=None):
        self.raise_fatal()
        try:
            return self._count_recorded(messages=messages, text=text, tools=tools, tool_choice=tool_choice)
        except (ReplayUnavailable, ReplayViolation) as exc:
            self.fatal = exc
            raise

    def raise_fatal(self):
        # The production controller catches candidate errors. Archive gaps
        # and replay mismatches must retain their audit classification even
        # when the controller catches them and continues to another view.
        if self.fatal is not None:
            raise self.fatal

    def _count_recorded(self, *, messages=None, text=None, tools=None, tool_choice=None):
        self._no_tools(tools, tool_choice)
        body = self._common(messages=messages, text=text)
        key = object_hash(body)
        if self.count_cursor >= len(self.expected_counts):
            raise ReplayUnavailable('missing_archived_exact_count_event')
        entry = self.expected_counts[self.count_cursor]
        require(entry['input_sha256'] == key, 'exact_count_event_order_or_input_hash_mismatch')
        require(entry.get('model') == self.model, 'recorded_count_model_mismatch')
        self.count_cursor += 1
        last_attempt = self._audit_physical_count_attempts(body, entry) if self.phase == 'recovery' else None
        if entry.get('status') != 'ok':
            if (last_attempt and last_attempt.get('generic_network_failure') is True
                    and entry.get('error_type') == 'OpenAIResponsesTransportError'):
                raise OpenAIResponsesTransportError('OpenAI request failed; response or usage may be unknown')
            # Count audit stores only the exception class, not its original
            # sanitized message. Do not invent an exact error replay.
            raise ReplayUnavailable('recorded_count_failure_lacks_exception_payload')
        require(key in self.count_map, 'exact_count_map_entry_missing')
        return self.count_map[key]

    def _audit_physical_count_attempts(self, body, logical):
        if not isinstance(self.expected_count_attempts, list):
            raise ReplayUnavailable('recovery_physical_count_attempt_audit_missing')
        if logical.get('cache_hit') is True:
            require(logical.get('status') == 'ok', 'cached_count_event_reported_failure')
            return None
        require(logical.get('cache_hit') is False, 'count_cache_flag_missing')
        key = text_hash(json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(',', ':')))
        operation_attempts = 0
        while operation_attempts < 3:
            if self.attempt_cursor >= len(self.expected_count_attempts):
                raise ReplayUnavailable('physical_count_attempt_group_incomplete')
            attempt = self.expected_count_attempts[self.attempt_cursor]
            operation_attempts += 1
            require(attempt.get('request_sha256') == key and attempt.get('model') == self.model,
                    'physical_count_attempt_input_or_model_mismatch')
            require(attempt.get('attempt') == operation_attempts, 'physical_count_retry_order_mismatch')
            number = attempt.get('physical_count_request')
            require(type(number) is int and number > 0, 'physical_count_identity_invalid')
            if self.attempt_cursor:
                require(number == self.expected_count_attempts[self.attempt_cursor-1]['physical_count_request']+1,
                        'physical_count_requests_not_sequential_within_task')
            self.attempt_cursor += 1
            if attempt.get('status') == 'ok':
                if logical.get('status') != 'ok':
                    raise ReplayUnavailable('count_payload_validation_failed_but_original_payload_not_retained')
                require(logical.get('status') == 'ok' and attempt.get('input_tokens') == logical.get('input_tokens'),
                        'logical_and_physical_count_result_mismatch')
                require('retry_wait_seconds' not in attempt, 'successful_count_attempt_has_retry_delay')
                break
            if attempt.get('status') != 'error':
                raise ReplayUnavailable('count_attempt_remained_inflight_or_unknown')
            retry = attempt.get('generic_network_failure') is True and operation_attempts < 3
            if retry:
                require(attempt.get('retry_wait_seconds') == (5 if operation_attempts == 1 else 15),
                        'count_retry_delay_differs_from_frozen_protocol')
                continue
            require('retry_wait_seconds' not in attempt and logical.get('status') == 'error',
                    'nonretryable_or_final_count_failure_has_inconsistent_outcome')
            break
        if operation_attempts > 1:
            self.retried_count_operations += 1
        return attempt

    def request(self, path, body=None, *, outside_v1=False):
        self.raise_fatal()
        try:
            return self._request_recorded(path, body, outside_v1=outside_v1)
        except (ReplayUnavailable, ReplayViolation) as exc:
            self.fatal = exc
            raise

    def _request_recorded(self, path, body=None, *, outside_v1=False):
        require(path == '/chat/completions' and not outside_v1, 'unexpected_replay_request_path')
        require(isinstance(body, dict), 'missing_replay_request_body')
        self._no_tools(body.get('tools'), body.get('tool_choice'))
        require(body.get('model') == self.model, 'replayed_request_model_mismatch')
        if self.generation_cursor >= len(self.expected_ledger):
            raise ReplayUnavailable('missing_archived_generation_entry')
        entry = self.expected_ledger[self.generation_cursor]
        number = self.generation_cursor + 1
        if not isinstance(self.expected_responses, list) or len(self.expected_responses) != len(self.expected_ledger):
            raise ReplayUnavailable('archived_response_wire_audit_missing_or_incomplete')
        archived_wire = self.expected_responses[self.generation_cursor]
        require(entry.get('physical_request') == number, 'archived_generation_order_mismatch')
        require(object_hash(body.get('messages')) == entry.get('prompt_sha256'), 'wire_prompt_sha256_mismatch')
        require(body.get('max_tokens') == entry.get('requested_output_tokens'), 'output_reservation_mismatch')
        require(not entry.get('native_tools') and entry.get('tools_sha256') is None, 'unexpected_archived_native_tools')
        self.wire_prompt_hashes_checked += 1
        wire = self._common(messages=body['messages'])
        wire.update(store=False, service_tier='default', max_output_tokens=body['max_tokens'])
        metadata = deepcopy(entry.get('provider_metadata') or {})
        if not archived_wire.get('request_sha256'):
            raise ReplayUnavailable('archived_response_wire_hash_missing')
        require(object_hash(wire) == archived_wire['request_sha256'], 'responses_wire_request_sha256_mismatch')
        require(archived_wire.get('model') == self.model, 'archived_response_wire_model_mismatch')
        if metadata.get('request_sha256'):
            require(metadata['request_sha256'] == archived_wire['request_sha256'], 'response_and_ledger_wire_hash_mismatch')
        self.wire_request_hashes_checked += 1
        user = next((m['content'] for m in reversed(body['messages']) if m.get('role') == 'user'), '')
        kind = ('verifier' if user.startswith('You are a strict evidence verifier for a multiple-choice question.')
                else 'candidate' if user.startswith('You are answering a multiple-choice question after an evidence-aware')
                else 'direct' if user.startswith('Please read the following text and answer the question below.')
                else 'other')
        self.kind_counts[kind] += 1
        self.generation_cursor += 1
        if entry.get('usage_known'):
            usage = deepcopy(entry.get('usage') or {})
            require(all(type(usage.get(k)) is int and usage[k] >= 0
                        for k in ('prompt_tokens', 'completion_tokens')), 'known_usage_missing_archived_counters')
            require(metadata.get('actual_model') == self.model, 'archived_actual_model_mismatch')
            self.known_responses += 1
            if entry.get('status') == 'error':
                if entry.get('error_type') == 'OpenAIResponseNotCompletedError':
                    raise OpenAIResponseNotCompletedError(usage=usage, metadata=metadata)
                raise ReplayUnavailable('known_noncompleted_error_not_reconstructable_from_archive')
            require(entry.get('status') == 'ok' and isinstance(entry.get('content'), str), 'completed_response_content_missing')
            message = {'role': 'assistant', 'content': entry['content'], 'openai_response': metadata}
            return {'choices': [{'message': message, 'finish_reason': entry.get('finish_reason')}],
                    'usage': usage, 'openai_response': metadata}
        self.unknown_errors += 1
        exception_types = {
            'OpenAIResponsesTransportError': OpenAIResponsesTransportError,
            'BudgetIntegrityError': BudgetIntegrityError, 'BudgetExhausted': BudgetExhausted,
            'ValueError': ValueError, 'RuntimeError': RuntimeError, 'TimeoutError': TimeoutError,
            'ConnectionError': ConnectionError, 'OSError': OSError,
        }
        kind = entry.get('error_type')
        if kind not in exception_types or not isinstance(entry.get('error'), str):
            raise ReplayUnavailable('unknown_usage_error_lacks_reconstructable_exception')
        raise exception_types[kind](entry['error'])

    def assert_consumed(self):
        self.raise_fatal()
        require(self.generation_cursor == len(self.expected_ledger), 'archived_generation_entries_not_consumed_once')
        require(self.count_cursor == len(self.expected_counts), 'archived_count_events_not_consumed_once')
        if self.phase == 'recovery':
            require(self.attempt_cursor == len(self.expected_count_attempts), 'physical_count_attempts_not_consumed_once')


def evidence_bindings(groups, context):
    """Parse actual renderer headers, skipping over complete source snippets."""
    records = []
    span_union = set()
    for group in groups:
        text = str(group.get('text') or '')
        cursor, checked, snippet_chars = 0, 0, 0
        while cursor < len(text):
            if not text[cursor:].strip():
                cursor = len(text)
                break
            header = HEADER.search(text, cursor)
            require(header is not None, 'rendered_evidence_header_missing')
            between = text[cursor:header.start()]
            require(not between.strip() or re.fullmatch(r'(?:\s*\[narrative \d+\]\s*)+', between) is not None,
                    'unexpected_unbound_text_between_evidence_windows')
            start, end = map(int, header.groups())
            require(0 <= start < end <= len(context), 'rendered_source_span_outside_context')
            snippet_end = header.end() + end - start
            require(text[header.end():snippet_end] == context[start:end], 'rendered_evidence_differs_from_exact_source_slice')
            cursor = snippet_end
            checked += 1
            snippet_chars += end-start
            span_union.add((start, end))
        records.append({'candidate_index': group['candidate_index'], 'query_mode': group['query_mode'],
                        'evidence_view_sha256': text_hash(text), 'rendered_source_spans_checked': checked,
                        'rendered_source_characters_checked': snippet_chars})
    return {'views': records, 'source_bindings_checked': sum(r['rendered_source_spans_checked'] for r in records),
            'source_characters_checked_with_reuse': sum(r['rendered_source_characters_checked'] for r in records),
            'unique_rendered_source_spans': len(span_union), 'source_slice_binding_verified': True}


def independent_verifier_span_counts(result, groups):
    """Recompute the frozen span-backed status counts without its aggregator."""
    counts = {letter: {'support': 0, 'against': 0, 'unknown': 0, 'span_backed': 0} for letter in 'ABCD'}
    valid_mentions = invalid_mentions = unparseable_mentions = 0
    evidence = [str(group.get('text') or '') for group in groups]
    if len(result.get('candidate_results') or []) == len(evidence)+1 and evidence:
        evidence.append(evidence[min(2, len(evidence)-1)])
    headers = [{tuple(map(int, match)) for match in re.findall(r'source_chars\s+(\d+)\s*:\s*(\d+)', text)}
               for text in evidence]
    for item in result.get('option_verifications') or []:
        index = item.get('candidate_index')
        require(type(index) is int and 0 <= index < len(headers), 'verifier_candidate_index_not_bound_to_view')
        for letter in 'ABCD':
            option = (item.get('options') or {}).get(letter) or {}
            parsed = []
            for span in option.get('evidence_spans') or []:
                match = re.search(r'(?:source_chars\s+)?(\d+)\s*:\s*(\d+)', str(span))
                if match:
                    value = tuple(map(int, match.groups()))
                    parsed.append(value)
                    if value in headers[index]:
                        valid_mentions += 1
                    else:
                        invalid_mentions += 1
                else:
                    unparseable_mentions += 1
            backed = bool(parsed) and all(span in headers[index] for span in parsed)
            status = option.get('status', 'unknown')
            effective = status if status in {'support', 'against'} and backed else 'unknown'
            counts[letter][effective] += 1
            counts[letter]['span_backed'] += int(backed and effective != 'unknown')
    aggregate = result.get('option_verification')
    if aggregate is not None:
        require(aggregate.get('status_counts') == counts, 'independent_verifier_span_status_counts_mismatch')
    return {'span_backed_status_counts_verified': aggregate is not None,
            'verifier_span_mentions_in_retrieved_headers': valid_mentions,
            'verifier_span_mentions_outside_retrieved_headers': invalid_mentions,
            'unparseable_verifier_span_mentions': unparseable_mentions,
            'policy_note': 'Only parseable spans are eligible under the frozen policy; unbacked claims are downgraded to unknown.'}


def stable_result(value):
    result = deepcopy(value)
    result.pop('seconds', None)
    if isinstance(result.get('shared_usage'), dict):
        result['shared_usage'].pop('model_seconds', None)
    return result


def compare_ledgers(actual, expected):
    fields = ('physical_request', 'prompt_sha256', 'tools_sha256', 'native_tools',
              'admission_prompt_tokens', 'requested_output_tokens', 'charged_tokens',
              'status', 'usage_known', 'finish_reason', 'empty_content', 'reasoning_present',
              'content', 'tool_calls', 'error_type', 'error')
    require(len(actual) == len(expected), 'replayed_ledger_length_mismatch')
    for left, right in zip(actual, expected):
        require({key: left.get(key) for key in fields} == {key: right.get(key) for key in fields},
                'replayed_ledger_entry_mismatch')
        if right.get('usage_known'):
            require(left.get('usage') == right.get('usage'), 'replayed_known_usage_mismatch')


def replay_record(record, value, groups_cache, *, phase='original', private_choices=None):
    if (record.get('error') or {}).get('type') in {'InterruptedTask', 'ModelBlocked', 'CostCapExceeded'}:
        raise ReplayUnavailable('terminal_control_error_has_no_replayable_complete_trajectory')
    if any(item.get('type') == 'CostCapExceeded' for name in ('candidate_errors', 'option_verification_errors')
           for item in (record.get('result') or {}).get(name, [])):
        raise ReplayUnavailable('global_monetary_denial_not_replayable_from_per_task_archive')
    row = LongBenchRow.from_mapping(value, offset=value['index'])
    require(row.answer == '', 'label_reached_trajectory_replay')
    transport = ReplayTransport(record, phase=phase)
    client = OpenAIResponsesBudgetedClient(transport, ExperimentBudget(**record['budget']))
    capped = _CallRecordingCostGate(client, f"{record['arm']}:{record['index']}")
    prompt = official_prompt(row)
    if record['arm'] == 'capacity_probe':
        raw = transport.count(messages=[{'role': 'user', 'content': prompt}])
        require(record['budget']['context_tokens'] == raw+1024+4096
                and record['budget']['total_tokens'] == raw+1024+4096,
                'probe_submission_budget_differs_from_registered_formula')
    source = transport.count(text=row.context)
    admission = client.count_messages([{'role': 'user', 'content': prompt}])
    measurement = record.get('measurements') or {}
    require(measurement.get('source_envelope_input_tokens') == source
            and measurement.get('full_prompt_admission_input_tokens') == admission,
            'source_or_admission_measurement_mismatch')
    require(measurement.get('fits_native_direct') is (admission+1024 <= NATIVE), 'native_capacity_classification_mismatch')
    if record['arm'] == 'capacity_probe':
        require(admission+1024 > NATIVE, 'capacity_probe_does_not_exceed_native_window')
    if record['status'] == 'provider_context_rejected':
        require(record['arm'] == 'capacity_probe'
                and 'status 400; code context_length_exceeded' in str((record.get('error') or {}).get('message', '')),
                'provider_context_rejection_status_not_backed_by_recorded_error_code')
    binding, span_counts, groups = None, None, None
    content, result, observed_exception = '', {}, None
    try:
        if record['status'] == 'native_admission_denied':
            require(record['arm'] == 'direct' and admission+1024 > NATIVE and not record['ledger'],
                    'invalid_native_admission_denial')
        elif record['arm'] == 'harness':
            if not record.get('result'):
                raise ReplayUnavailable('terminal_harness_error_has_no_returned_controller_artifact')
            if value['index'] not in groups_cache:
                cfg = LongBenchEvidenceConfig()
                print(json.dumps({'offline_audit': 'regenerating_source_views', 'index': value['index'],
                                  'context_chars': len(row.context)}), flush=True)
                groups = evidence_window_groups(row, chunk_items(row.context, cfg.chunk_chars),
                    candidate_count=cfg.candidate_count, max_windows=cfg.max_windows,
                    window_chars=cfg.window_chars, max_total_chars=cfg.evidence_max_chars)
                groups_cache[value['index']] = groups
            groups = groups_cache[value['index']]
            binding = evidence_bindings(groups, row.context)
            result = run_longbench_harness(row, capped, groups=groups)
            transport.raise_fatal()
            content = str(result.get('prediction') or '')
            require(stable_result(result) == stable_result(record['result']), 'controller_result_vote_or_verification_mismatch')
            span_counts = independent_verifier_span_counts(result, groups)
        else:
            content = capped.complete([{'role': 'user', 'content': prompt}], max_tokens=1024)
    except (ReplayUnavailable, ReplayViolation):
        raise
    except Exception as exc:
        observed_exception = type(exc).__name__
        require(observed_exception == (record.get('error') or {}).get('type'), 'replayed_terminal_exception_type_mismatch')
    prediction = (content if record['arm'] == 'harness' and len(content) == 1 and content in 'ABCD'
                  else extract_choice(content))
    require(prediction == record.get('prediction'), 'replayed_final_prediction_mismatch')
    transport.assert_consumed()
    compare_ledgers(client.ledger, record['ledger'])
    receipt = strict_choice_receipt(record, row, groups, capped.call_events)
    if private_choices is not None:
        private_choices.append(receipt)
    return {'status': 'passed', 'count_events_replayed': transport.count_cursor,
            'physical_count_attempts_verified': transport.attempt_cursor,
            'recovery_count_operations_retried': transport.retried_count_operations,
            'generation_entries_replayed': transport.generation_cursor,
            'known_responses_replayed': transport.known_responses,
            'unknown_delivery_errors_replayed': transport.unknown_errors,
            'wire_prompt_hashes_checked': transport.wire_prompt_hashes_checked,
            'responses_wire_request_hashes_checked': transport.wire_request_hashes_checked,
            'generation_kind_counts': dict(transport.kind_counts),
            'prediction_matches_stored': True, 'ledger_matches_stored': True,
            'controller_result_matches_stored': record['arm'] == 'harness',
            'observed_terminal_exception_type': observed_exception,
            'source_views': binding, 'verifier_span_counts': span_counts,
            'official_raw_parser': {
                'raw_reply_available': receipt['raw_reply_available'],
                'raw_reply_sha256': receipt['raw_reply_sha256'],
                'official_choice_available': receipt['strict_choice'] is not None,
                'local_raw_choice_available': receipt['local_raw_reply_choice'] is not None,
                'official_and_local_raw_parser_agree': receipt['official_and_local_raw_parser_agree'],
                'official_and_recorded_primary_agree': receipt['official_and_recorded_primary_agree'],
                'selected_provenance': receipt['selected_provenance']},
            'controller_counter_matches_physical_attempts': (
                (result.get('model_calls') == len(client.ledger)) if result else None)}


def classify_record(record):
    arm, status = record['arm'], record['status']
    if zero_generation_network_failure(record):
        return 'zero_generation_counting_infrastructure_failure'
    if arm == 'capacity_probe':
        return ('physical_provider_context_rejection_probe' if status == 'provider_context_rejected'
                else 'unexpected_probe_return' if status == 'probe_unexpectedly_returned'
                else 'probe_execution_error_not_capacity_measurement')
    if status == 'native_admission_denied':
        return 'native_capacity_admission_denial_no_generation'
    if status == 'returned':
        return 'returned_harness_answer' if arm == 'harness' else 'returned_native_direct_answer'
    if status == 'no_parseable_answer':
        return 'controller_returned_without_parseable_answer'
    return 'terminal_execution_or_control_error'


def safe_report(value):
    forbidden = {'context', 'question', 'answer', 'content', 'prediction', 'reasoning',
                 'evidence_text', 'api_key', 'headers', 'authorization'}
    def visit(item):
        if isinstance(item, dict):
            require(not forbidden.intersection(item), 'raw_or_secret_field_in_audit_report')
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)
        elif isinstance(item, str):
            require(not re.search(r'\bsk-[A-Za-z0-9_-]{12,}|/(?:Users|home|uniparser)/', item), 'private_string_in_audit_report')
    visit(value)


def zero_generation_network_failure(record):
    usage, error = record.get('usage') or {}, record.get('error') or {}
    return (record.get('terminal') is True and record.get('status') == 'error'
            and record.get('ledger') == [] and usage.get('physical_requests') == 0
            and usage.get('charged_tokens') == 0
            and error.get('type') == 'OpenAIResponsesTransportError'
            and error.get('message') == 'OpenAI request failed; response or usage may be unknown')


def audit(formal, *, recovery=None, control=None, authorized_after_all_114=False, private_choices=None):
    if not authorized_after_all_114:
        raise ReplayUnavailable('parent_authorization_after_all_114_required')
    formal = Path(formal).resolve()
    recovery = Path(recovery).resolve() if recovery else formal.parent/'recovery'
    control = Path(control).resolve() if control else formal.parent/'headtail'
    # Small metadata gates precede any prediction/source read.
    run = load(formal/'run_summary.json')
    require(run.get('phase') == 'inference_complete' and run.get('terminal_records') == 75
            and run.get('planned_main_records') == 72 and run.get('planned_probe_records') == 3,
            'all_75_terminal_gate_not_satisfied')
    child_run = load(recovery/'run_summary.json')
    require(child_run.get('phase') == 'recovery_inference_complete'
            and child_run.get('planned_recovery_records') == 39
            and child_run.get('terminal_recovery_records') == 39
            and child_run.get('parent_terminal_records') == 75,
            'all_39_recovery_terminal_gate_not_satisfied')
    child_access = load(recovery/'scoring_access.json')
    require(child_access.get('all_75_parent_and_39_recovery_terminal_before_labels') is True,
            'combined_114_before_label_access_gate_not_satisfied')
    require((recovery/'audit.json').is_file() and (formal/'audit.json').is_file(),
            'unified_composite_scoring_not_finished')
    access = load(formal/'scoring_access.json')
    require(access.get('all_75_terminal_before_label_access') is True, 'unified_scoring_gate_not_satisfied')
    parser_source = official_parser_provenance()
    choice_receipts = private_choices if private_choices is not None else []
    frozen = load(formal/'freeze.json')
    require(digest(formal/'freeze.json') == EXPECTED_FREEZE, 'unexpected_experiment_freeze')
    child_frozen = load(recovery/'freeze.json')
    require(digest(recovery/'freeze.json') == EXPECTED_RECOVERY_FREEZE, 'unexpected_recovery_freeze')
    lineage = child_frozen.get('recovery') or {}
    control_frozen, control_access = control_metadata_gate(control, frozen, child_frozen)
    require(control_frozen['registered_at_unix'] < min(access['first_scored_at_unix'],
        child_access['first_scored_at_unix'], control_access['first_scored_at_unix']),
        'headtail_registration_did_not_precede_score_access_metadata')
    require(lineage.get('parent_freeze_sha256') == EXPECTED_FREEZE
            and lineage.get('expected_parent_records') == 75
            and lineage.get('expected_recovery_records') == 39,
            'recovery_parent_or_matrix_registration_mismatch')
    require(child_frozen['code_sha256'] == frozen['code_sha256']
            and lineage.get('parent_code_sha256') == frozen['code_sha256'],
            'recovery_changed_original_controller_source_set')
    require(child_frozen['inputs_sha256'] == frozen['inputs_sha256']
            and child_frozen['labels_sha256'] == frozen['labels_sha256']
            and child_access.get('labels_sha256') == frozen['labels_sha256'],
            'recovery_changed_source_or_declared_label_identity')
    require(len(frozen['code_sha256']) == 56, 'unexpected_frozen_source_file_count')
    for name, expected in frozen['code_sha256'].items():
        path = Path(name)
        require(not path.is_absolute() and '..' not in path.parts and path.suffix == '.py', 'invalid_frozen_source_path')
        require(digest(ROOT/path) == expected, 'live_source_hash_differs_from_freeze')
        require(digest(formal/'frozen_source'/path) == expected, 'snapshot_source_hash_differs_from_freeze')
        require(digest(recovery/'frozen_source'/path) == expected, 'recovery_snapshot_source_hash_differs_from_freeze')
    wrapper = 'scripts/benchmark_longbench_openai_recovery.py'
    require(digest(ROOT/wrapper) == lineage.get('wrapper_sha256')
            and digest(recovery/'frozen_source'/wrapper) == lineage.get('wrapper_sha256'),
            'recovery_wrapper_live_or_snapshot_changed')
    config = frozen['config']
    require(digest(config['manifest']) == frozen['manifest_sha256'], 'parent_selection_manifest_hash_mismatch')
    parent_manifest = load(config['manifest'])
    require(digest(config['inputs']) == frozen['inputs_sha256'], 'frozen_inputs_file_hash_mismatch')
    values = [json.loads(line) for line in Path(config['inputs']).read_text().splitlines() if line.strip()]
    require(len(values) == 12 and len({v['index'] for v in values}) == 12, 'source_input_matrix_incomplete')
    rows = {}
    for value in values:
        require(not {'answer', 'answers', 'gold', 'labels', 'correct'} & value.keys(), 'source_input_has_labels')
        key = str(value['index'])
        require(value['_id'] == frozen['sample_ids'][key], 'source_input_identity_mismatch')
        require(text_hash(value['context']) == frozen['source_hashes'][key], 'source_input_context_hash_mismatch')
        rows[value['index']] = value
    control_preparation, control_values, control_metadata = audit_headtail_preparation(
        control, control_frozen, rows, parent_manifest)
    expected = {(model, index, arm) for model in MODELS for index in rows for arm in ('direct', 'harness')}
    expected.update((model, frozen['probe_index'], 'capacity_probe') for model in MODELS)
    paths = list((formal/'predictions').glob('*.json'))
    require(len(paths) == 75, 'terminal_record_file_count_mismatch')
    original_by_name = {path.name: load(path) for path in paths}
    original_hashes = {path.name: digest(path) for path in paths}
    require(original_hashes == lineage.get('parent_record_sha256'), 'original_75_records_changed_after_recovery_freeze')
    require({(r.get('model'), r.get('index'), r.get('arm')) for r in original_by_name.values()} == expected,
            'terminal_record_identity_matrix_mismatch')
    eligible_names = {name for name, record in original_by_name.items() if zero_generation_network_failure(record)}
    require(len(eligible_names) == 39, 'recovery_eligibility_count_not_39')
    jobs = lineage.get('execution_order') or []
    expected_recovery = {(job['model'], job['index'], job['arm']) for job in jobs}
    require(len(jobs) == len(expected_recovery) == 39
            and expected_recovery == {(original_by_name[name]['model'], original_by_name[name]['index'],
                                      original_by_name[name]['arm']) for name in eligible_names},
            'registered_recovery_subset_differs_from_metadata_only_eligibility')
    child_paths = list((recovery/'predictions').glob('*.json'))
    require(len(child_paths) == 39 and {path.name for path in child_paths} == eligible_names,
            'recovery_record_file_matrix_mismatch')
    recovered_by_name = {path.name: load(path) for path in child_paths}
    recovered_hashes = {path.name: digest(path) for path in child_paths}
    require(recovered_hashes == control_frozen['recovery_record_sha256'],
            'recovery_records_changed_after_headtail_registration')
    require(digest(formal/'cost_ledger.json') == control_frozen['parent_cost_ledger_sha256']
            and digest(recovery/'cost_ledger.json') == control_frozen['recovery_cost_ledger_sha256'],
            'preceding_cost_history_changed_after_headtail_registration')
    for name, record in recovered_by_name.items():
        parent = original_by_name[name]
        require(record.get('terminal') is True and record.get('freeze_sha256') == EXPECTED_RECOVERY_FREEZE
                and record.get('parent_freeze_sha256') == EXPECTED_FREEZE
                and record.get('parent_record_sha256') == original_hashes[name],
                'recovery_record_parent_lineage_mismatch')
        require(all(record.get(key) == parent.get(key) for key in ('model', 'index', 'arm', 'id', 'context_sha256')),
                'recovery_record_changed_logical_case_identity')
        require(record.get('logical_parent_physical_requests') == 0
                and record.get('logical_parent_charged_tokens') == 0,
                'recovery_claims_nonzero_parent_budget')
    records = list(original_by_name.values()) + list(recovered_by_name.values())
    cache, results = {}, []
    totals = Counter()
    violations = []
    started = time.perf_counter()
    for record in sorted(records, key=lambda r: (r['index'], r['model'], r['arm'], r['freeze_sha256'])):
        phase = 'recovery' if record['freeze_sha256'] == EXPECTED_RECOVERY_FREEZE else 'original'
        name = f"{record['index']}__{record['model']}__{record['arm']}.json"
        selected = phase == 'recovery' or name not in eligible_names
        identity = {'index': record['index'], 'id': record['id'], 'model': record['model'], 'arm': record['arm'],
                    'phase': phase, 'actual_record_freeze_sha256': record['freeze_sha256'],
                    'logical_main_freeze_sha256': EXPECTED_FREEZE,
                    'original_record_sha256': original_hashes[name],
                    'actual_record_sha256': (recovered_hashes[name] if phase == 'recovery' else original_hashes[name]),
                    'context_sha256': record['context_sha256'],
                    'outcome_origin': ('zero_generation_network_recovery' if phase == 'recovery'
                                       else 'original_trial' if selected else 'preserved_first_attempt_failure'),
                    'selected_for_logical_outcome': selected}
        require(record.get('terminal') is True
                and record['freeze_sha256'] == (EXPECTED_RECOVERY_FREEZE if phase == 'recovery' else EXPECTED_FREEZE),
                'terminal_record_freeze_or_terminal_flag_mismatch')
        require(record['id'] == frozen['sample_ids'][str(record['index'])]
                and record['context_sha256'] == frozen['source_hashes'][str(record['index'])],
                'terminal_record_source_identity_mismatch')
        expected_budget = config['budget_harness' if record['arm'] == 'harness' else 'budget_direct']
        if record['arm'] == 'capacity_probe':
            require(set(record['budget']) == set(expected_budget)
                    and all(record['budget'][key] == value for key, value in expected_budget.items()
                            if key not in {'context_tokens', 'total_tokens'}),
                    'capacity_probe_fixed_budget_fields_changed')
        else:
            require(record['budget'] == expected_budget, 'task_budget_differs_from_registered_arm')
        ledger = audit_ledger(record['ledger'], record['budget'])
        ledger_ok = not ledger['violations']
        item = {**identity, 'terminal_status': record['status'], 'classification': classify_record(record),
                'ledger_passed': ledger_ok, 'ledger_violations': ledger['violations'],
                'physical_generation_attempts': ledger['physical_requests'],
                'known_input_tokens': ledger['prompt_tokens'], 'known_output_tokens': ledger['completion_tokens'],
                'charged_or_reserved_tokens': ledger['charged_tokens'],
                'unknown_usage_attempts': ledger['unknown_usage_requests'],
                'recorded_count_events': len(record.get('count_audit') or []),
                'recorded_count_errors': sum(entry.get('status') == 'error' for entry in record.get('count_audit') or []),
                'physical_count_attempt_journal_available': isinstance(record.get('count_attempt_audit'), list),
                'recorded_physical_count_attempts': (len(record['count_attempt_audit'])
                    if isinstance(record.get('count_attempt_audit'), list) else None),
                'recorded_provider_diagnostics': len(record.get('provider_error_diagnostics') or []),
                'recorded_terminal_error_type': (record.get('error') or {}).get('type')}
        if ledger_ok:
            totals['ledger_passed'] += 1
        else:
            violations.append({**identity, 'code': 'token_ledger_violations'})
        try:
            if phase == 'original' and name in eligible_names:
                raise ReplayUnavailable('preserved_original_zero_generation_infrastructure_failure')
            item['trajectory'] = replay_record(record, rows[record['index']], cache, phase=phase,
                                               private_choices=choice_receipts)
            totals['trajectory_passed'] += 1
            for key in ('generation_entries_replayed', 'known_responses_replayed', 'unknown_delivery_errors_replayed',
                        'count_events_replayed', 'physical_count_attempts_verified', 'recovery_count_operations_retried',
                        'wire_prompt_hashes_checked', 'responses_wire_request_hashes_checked'):
                totals[key] += item['trajectory'][key]
            binding = item['trajectory']['source_views']
            if binding:
                totals['harness_source_view_trajectories_passed'] += 1
                totals['source_bindings_checked'] += binding['source_bindings_checked']
                totals['source_view_characters_checked_with_reuse'] += binding['source_characters_checked_with_reuse']
        except ReplayUnavailable as exc:
            item['trajectory'] = {'status': 'unavailable', 'reason_code': exc.code}
            totals['trajectory_unavailable'] += 1
        except ReplayViolation as exc:
            item['trajectory'] = {'status': 'failed', 'reason_code': exc.code}
            totals['trajectory_failed'] += 1
            violations.append({**identity, 'code': exc.code})
        except Exception as exc:
            item['trajectory'] = {'status': 'unavailable', 'reason_code': 'unexpected_replay_exception',
                                  'exception_type': type(exc).__name__}
            totals['trajectory_unavailable'] += 1
        if selected and not any(receipt['index'] == record['index'] and receipt['model'] == record['model']
                                and receipt['arm'] == record['arm'] for receipt in choice_receipts):
            choice_receipts.append({'index': record['index'], 'id': record['id'], 'model': record['model'],
                'arm': record['arm'], 'record_status': record['status'],
                'actual_record_freeze_sha256': record['freeze_sha256'],
                'source_context_sha256': record['context_sha256'],
                'original_context_sha256': record.get('original_context_sha256', record['context_sha256']),
                'raw_reply_available': False, 'raw_reply_sha256': None,
                'strict_choice': None, 'local_raw_reply_choice': None,
                'actual_controller_choice': record.get('prediction'),
                'selected_provenance': {'kind': 'raw_reply_unavailable_due_to_inconclusive_or_failed_replay',
                    'reason_code': item['trajectory'].get('reason_code')},
                'official_and_local_raw_parser_agree': None,
                'official_and_recorded_primary_agree': None})
        totals['terminal_records'] += 1
        totals['physical_generation_attempts'] += ledger['physical_requests']
        totals['charged_or_reserved_tokens'] += ledger['charged_tokens']
        totals['unknown_usage_attempts'] += ledger['unknown_usage_requests']
        totals['recorded_count_events_all_phases'] += item['recorded_count_events']
        totals['recorded_count_error_events_all_phases'] += item['recorded_count_errors']
        if phase == 'recovery':
            totals['recorded_recovery_physical_count_attempts'] += item['recorded_physical_count_attempts'] or 0
        results.append(item)
        if len(results) % 10 == 0:
            print(json.dumps({'offline_audit': 'core_record_progress', 'audited_physical_records': len(results),
                              'trajectory_passed': totals['trajectory_passed'],
                              'trajectory_unavailable': totals['trajectory_unavailable'],
                              'trajectory_failed': totals['trajectory_failed']}), flush=True)
    logical = [item for item in results if item['selected_for_logical_outcome']]
    preserved = [item for item in results if not item['selected_for_logical_outcome']]
    require(len(results) == 114 and len(logical) == 75 and len(preserved) == 39,
            'physical_history_or_logical_overlay_size_mismatch')
    totals['logical_outcomes'] = len(logical)
    totals['logical_trajectories_passed'] = sum(item['trajectory']['status'] == 'passed' for item in logical)
    totals['logical_trajectories_unavailable'] = sum(item['trajectory']['status'] == 'unavailable' for item in logical)
    totals['preserved_original_zero_generation_failures'] = len(preserved)
    control_results = []
    control_paths = list((control/'predictions').glob('*.json'))
    expected_control = {(model, index) for model in MODELS for index in control_frozen['planned_indices']}
    control_records = [load(path) for path in control_paths]
    require(len(control_records) == control_frozen['expected_control_records']
            and {(record.get('model'), record.get('index')) for record in control_records} == expected_control,
            'headtail_terminal_record_matrix_mismatch')
    for record in sorted(control_records, key=lambda value: (value['index'], value['model'])):
        index, model = record['index'], record['model']
        metadata = control_metadata[index]
        require(record.get('terminal') is True and record.get('arm') == 'headtail_control'
                and record.get('freeze_sha256') == EXPECTED_CONTROL_FREEZE
                and record.get('parent_freeze_sha256') == EXPECTED_FREEZE
                and record.get('recovery_freeze_sha256') == EXPECTED_RECOVERY_FREEZE,
                'headtail_terminal_freeze_or_arm_identity_mismatch')
        require(record.get('id') == metadata['id']
                and record.get('context_sha256') == metadata['trimmed_context_sha256']
                and record.get('original_context_sha256') == metadata['original_context_sha256']
                and record.get('source_spans') == metadata['source_spans']
                and record.get('budget') == control_frozen['budget'],
                'headtail_record_projection_or_budget_mismatch')
        ledger = audit_ledger(record['ledger'], control_frozen['budget'])
        item = {'index': index, 'id': record['id'], 'model': model, 'arm': 'headtail_control',
            'phase': 'headtail', 'actual_record_freeze_sha256': EXPECTED_CONTROL_FREEZE,
            'logical_main_freeze_sha256': EXPECTED_FREEZE,
            'context_sha256': record['context_sha256'], 'original_context_sha256': record['original_context_sha256'],
            'terminal_status': record['status'], 'classification': 'lossy_headtail_quality_control',
            'source_projection_verified': True, 'ledger_passed': not ledger['violations'],
            'ledger_violations': ledger['violations'], 'physical_generation_attempts': ledger['physical_requests'],
            'charged_or_reserved_tokens': ledger['charged_tokens'], 'unknown_usage_attempts': ledger['unknown_usage_requests'],
            'recorded_count_events': len(record.get('count_audit') or []),
            'recorded_count_errors': sum(entry.get('status') == 'error' for entry in record.get('count_audit') or []),
            'recorded_physical_count_attempts': len(record.get('count_attempt_audit') or [])}
        if not ledger['violations']:
            totals['ledger_passed'] += 1
        else:
            violations.append({'index': index, 'model': model, 'arm': 'headtail_control', 'code': 'token_ledger_violations'})
        try:
            item['trajectory'] = replay_headtail(record, control_values[index], metadata, choice_receipts)
            totals['auxiliary_trajectories_passed'] += 1
            totals['trajectory_passed'] += 1
            for key in ('generation_entries_replayed', 'known_responses_replayed', 'unknown_delivery_errors_replayed',
                        'count_events_replayed', 'physical_count_attempts_verified', 'recovery_count_operations_retried',
                        'wire_prompt_hashes_checked', 'responses_wire_request_hashes_checked'):
                totals[key] += item['trajectory'][key]
        except ReplayUnavailable as exc:
            item['trajectory'] = {'status': 'unavailable', 'reason_code': exc.code}
            totals['auxiliary_trajectories_unavailable'] += 1
            totals['trajectory_unavailable'] += 1
        except ReplayViolation as exc:
            item['trajectory'] = {'status': 'failed', 'reason_code': exc.code}
            totals['auxiliary_trajectories_failed'] += 1
            totals['trajectory_failed'] += 1
            violations.append({'index': index, 'model': model, 'arm': 'headtail_control', 'code': exc.code})
        except Exception as exc:
            item['trajectory'] = {'status': 'unavailable', 'reason_code': 'unexpected_replay_exception',
                                  'exception_type': type(exc).__name__}
            totals['auxiliary_trajectories_unavailable'] += 1
            totals['trajectory_unavailable'] += 1
        if not any(receipt['index'] == index and receipt['model'] == model and receipt['arm'] == 'headtail_control'
                   for receipt in choice_receipts):
            choice_receipts.append({'index': index, 'id': record['id'], 'model': model, 'arm': 'headtail_control',
                'record_status': record['status'], 'actual_record_freeze_sha256': EXPECTED_CONTROL_FREEZE,
                'source_context_sha256': record['context_sha256'], 'original_context_sha256': record['original_context_sha256'],
                'raw_reply_available': False, 'raw_reply_sha256': None, 'strict_choice': None,
                'local_raw_reply_choice': None, 'actual_controller_choice': record.get('prediction'),
                'selected_provenance': {'kind': 'raw_reply_unavailable_due_to_inconclusive_or_failed_replay',
                    'reason_code': item['trajectory'].get('reason_code')},
                'official_and_local_raw_parser_agree': None, 'official_and_recorded_primary_agree': None})
        totals['terminal_records'] += 1
        totals['physical_generation_attempts'] += ledger['physical_requests']
        totals['charged_or_reserved_tokens'] += ledger['charged_tokens']
        totals['unknown_usage_attempts'] += ledger['unknown_usage_requests']
        totals['recorded_count_events_all_phases'] += item['recorded_count_events']
        totals['recorded_count_error_events_all_phases'] += item['recorded_count_errors']
        totals['recorded_headtail_physical_count_attempts'] += item['recorded_physical_count_attempts']
        control_results.append(item)
    totals['auxiliary_quality_control_outcomes'] = len(control_results)
    totals['all_phase_records'] = len(results)+len(control_results)
    all_results = results+control_results
    for status in ('passed', 'unavailable', 'failed'):
        require(totals['trajectory_'+status] == sum(item['trajectory']['status'] == status for item in all_results),
                'cross_phase_trajectory_status_total_mismatch')
    require(sum(totals['trajectory_'+status] for status in ('passed', 'unavailable', 'failed'))
            == totals['terminal_records'] == len(all_results), 'cross_phase_terminal_total_mismatch')
    for key in ('generation_entries_replayed', 'known_responses_replayed', 'unknown_delivery_errors_replayed',
                'count_events_replayed', 'physical_count_attempts_verified', 'recovery_count_operations_retried',
                'wire_prompt_hashes_checked', 'responses_wire_request_hashes_checked'):
        require(totals[key] == sum(item['trajectory'].get(key, 0) for item in all_results),
                'cross_phase_replay_counter_mismatch_'+key)
    require(totals['generation_entries_replayed'] == totals['known_responses_replayed']
            + totals['unknown_delivery_errors_replayed'], 'replayed_generation_kind_partition_mismatch')
    require(totals['recorded_count_events_all_phases'] == sum(item['recorded_count_events'] for item in all_results)
            and totals['recorded_count_error_events_all_phases'] == sum(item['recorded_count_errors'] for item in all_results),
            'cross_phase_recorded_count_total_mismatch')
    all_paths = paths+child_paths+control_paths
    require(all(path.stat().st_mtime < control_access['first_scored_at_unix'] for path in all_paths),
            'a_prediction_artifact_did_not_precede_final_score_access_metadata')
    require((ROOT/'work/longbench_openai_limits_20261102/official_parser_provenance.json').stat().st_mtime
            < control_access['first_scored_at_unix'], 'official_parser_snapshot_metadata_postdates_score_access')
    report = {'schema_version': 1, 'experiment': 'LongBench v2 OpenAI limits offline trajectory audit',
        'freeze_sha256': EXPECTED_FREEZE, 'status': 'fail' if violations else 'pass_available_audits',
        'recovery_freeze_sha256': EXPECTED_RECOVERY_FREEZE,
        'headtail_freeze_sha256': EXPECTED_CONTROL_FREEZE,
        'all_75_original_plus_39_recovery_terminal_and_scored_before_audit': True,
        'all_registered_auxiliary_controls_terminal_and_scored_before_audit': True,
        'labels_or_scored_gold_objects_opened_by_auditor': False,
        'provider_or_tokenizer_calls_made': 0, 'historical_results_opened': False,
        'source_checks': {'parent_live_and_snapshot_files_verified': 56,
                          'same_recovery_snapshot_files_verified': 56, 'recovery_wrapper_files_verified': 1,
                          'headtail_live_and_snapshot_files_verified': 58,
                          'distinct_source_hashes_verified': 12, 'original_record_hashes_preserved': 75,
                          'inputs_sha256': frozen['inputs_sha256']},
        'audit_scope': ['independent sequential token-ledger checks',
                        'exact archived count-event/input-hash replay',
                        'budget-notice and Responses wire hashes reconstructed',
                        'complete source-backed evidence views regenerated',
                        'existing controller candidate votes and option verification replayed',
                        'independent source-slice bindings and verifier span-status counts',
                        'count-only recovery attempts and registered 5/15-second retry metadata',
                        'zero-generation eligibility and all original failed attempts preserved'],
        'shared_implementation': 'Production source selection, parsers, controller, and reducers are reused; this is deterministic replay with independent wire/source/ledger checks.',
        'semantic_entailment_or_label_truth_verified': False,
        'official_parser_provenance': parser_source,
        'strict_parser_selection_policy': 'Apply official extraction to the original fixed selected raw reply; never select a candidate using the strict parser.',
        'strict_parser_summary': {
            'logical_outcomes_with_receipts': len(choice_receipts),
            'raw_replies_available': sum(receipt['raw_reply_available'] for receipt in choice_receipts),
            'official_choice_available': sum(receipt['strict_choice'] is not None for receipt in choice_receipts),
            'strict_vs_local_raw_disagreements': sum(receipt.get('official_and_local_raw_parser_agree') is False
                                                   for receipt in choice_receipts),
            'strict_vs_recorded_primary_disagreements': sum(receipt.get('official_and_recorded_primary_agree') is False
                                                          for receipt in choice_receipts),
            'gold_used_by_this_auditor': False},
        'unavailable_policy': 'Missing response/count/error payloads or unavailable controller artifacts are inconclusive, never passes.',
        'totals_scope': 'All 120 physical task histories, including 6 auxiliary headtail controls; logical_* covers the 75 main/probe overlay only. Preparation-only counts are separately reported in headtail_preparation_replay and are excluded from task totals.',
        'totals': dict(totals), 'classifications': dict(Counter(r['classification'] for r in results)),
        'records': results+control_results, 'logical_records': logical,
        'auxiliary_headtail_records': control_results, 'headtail_preparation_replay': control_preparation,
        'preserved_first_attempt_failures': preserved,
        'recovery_eligibility_used_predictions_or_scores': False,
        'recovery_retry_delays_reexecuted': False,
        'violations': violations,
        'auditor_sha256': digest(Path(__file__)), 'seconds': round(time.perf_counter()-started, 4)}
    safe_report(report)
    return report


def self_test():
    """One synthetic fixture only; no experiment paths or real credentials."""
    from collections import deque
    class FixtureTransport(LongBenchResponsesTransport):
        def __init__(self):
            super().__init__(api_key='OFFLINE_SYNTHETIC_FIXTURE', model=MODELS[2])
            self.count_attempt_audit = []
            unknown = json.dumps({'options': {letter: {'status': 'unknown', 'evidence_spans': [], 'reason': ''}
                                             for letter in 'ABCD'}, 'recommended': ''})
            self.replies = deque(['(A)', '(A)', 'The correct answer is (A)']+[unknown]*3)
        def _post(self, path, body):
            content = body['input']
            if isinstance(content, str):
                length = len(content)
            else:
                length = sum(len(message['content']) if isinstance(message['content'], str)
                             else sum(len(part['text']) for part in message['content']) for message in content)
            count = length//4+10
            if path == '/responses/input_tokens':
                key = text_hash(json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(',', ':')))
                retry = not self.count_attempt_audit
                if retry:
                    self.count_attempt_audit.append({'physical_count_request': 1,
                        'request_sha256': key, 'attempt': 1, 'status': 'error', 'model': self.model,
                        'generic_network_failure': True, 'error_type': 'OpenAIResponsesTransportError',
                        'retry_wait_seconds': 5})
                self.count_attempt_audit.append({'physical_count_request': len(self.count_attempt_audit)+1,
                    'request_sha256': key, 'attempt': 2 if retry else 1,
                    'status': 'ok', 'input_tokens': count, 'model': self.model})
                return {'object': 'response.input_tokens', 'input_tokens': count}
            reply = self.replies.popleft()
            return {'id': 'resp_fixture_'+str(len(self.replies)), 'model': self.model,
                'status': 'completed', 'service_tier': 'default',
                'usage': {'input_tokens': count, 'output_tokens': 30, 'total_tokens': count+30},
                'output': [{'type': 'message', 'role': 'assistant', 'status': 'completed',
                            'content': [{'type': 'output_text', 'text': reply}]}]}
    value = {'_id': 'fixture', 'id': 'fixture', 'index': 0, 'domain': 'Single-Document QA',
             'question': 'What is the flag color?', 'choice_A': 'red', 'choice_B': 'blue',
             'choice_C': 'green', 'choice_D': 'yellow', 'context': 'The flag is red. The town archive documents it.\n'*5000}
    row = LongBenchRow.from_mapping(value, offset=0)
    transport = FixtureTransport()
    limits = {'context_tokens': 16_384, 'total_tokens': 65_536, 'max_calls': 32,
              'max_output_tokens': 2048, 'token_safety_margin': 16}
    client = OpenAIResponsesBudgetedClient(transport, ExperimentBudget(**limits))
    capped = CostCappedClient(client, _NoopMonetaryGate(), 'harness:0')
    source = transport.count(text=row.context)
    admission = client.count_messages([{'role': 'user', 'content': official_prompt(row)}])
    result = run_longbench_harness(row, capped)
    record = {'model': transport.model, 'index': 0, 'arm': 'harness', 'status': 'returned', 'error': None,
        'budget': limits, 'result': result, 'prediction': result['prediction'],
        'measurements': {'source_envelope_input_tokens': source, 'full_prompt_admission_input_tokens': admission,
                         'fits_native_direct': admission+1024 <= NATIVE},
        'ledger': client.ledger, 'count_audit': transport.count_audit, 'response_audit': transport.response_audit,
        'count_attempt_audit': transport.count_attempt_audit}
    choices = []
    replay = replay_record(record, value, {}, phase='recovery', private_choices=choices)
    require(replay['status'] == 'passed' and replay['generation_entries_replayed'] == 6
            and replay['recovery_count_operations_retried'] == 1,
            'synthetic_self_test_replay_failed')
    require(official_strict_extract('The correct answer is (B). The correct answer is (A).') == 'B'
            and extract_choice('The correct answer is (B). The correct answer is (A).') == 'A',
            'official_first_match_policy_fixture_failed')
    require(official_strict_extract('**The correct answer is (D)**') == 'D'
            and official_strict_extract('the correct answer is (C)') is None
            and official_strict_extract('(A)') is None, 'official_case_or_format_policy_fixture_failed')
    require(len(choices) == 1 and choices[0]['selected_provenance'].get('strict_parser_used_to_select_candidate') is False,
            'strict_parser_changed_candidate_selection')
    require(choices[0]['selected_provenance'].get('candidate_index') in (0, 1)
            and choices[0]['strict_choice'] is None and choices[0]['local_raw_reply_choice'] == 'A',
            'strict_parser_must_not_replace_unformatted_selected_candidate_with_a_formatted_alternative')
    unicode_source = '前🙂\n中间ABCDEFG尾巴🚀'
    projection, spans, gap = headtail_projection(unicode_source, 9)
    require(projection == unicode_source[:5]+unicode_source[-4:]
            and spans == [[0, 5], [len(unicode_source)-4, len(unicode_source)]]
            and gap == [5, len(unicode_source)-4], 'synthetic_headtail_unicode_projection_failed')
    trimmed, spans, gap = headtail_projection(value['context'], 1000)
    control_value = {**value, 'context': trimmed}
    control_row = LongBenchRow.from_mapping(control_value, offset=0)
    control_transport = FixtureTransport()
    control_limits = {'context_tokens': 800_000, 'total_tokens': 800_000, 'max_calls': 1,
                      'max_output_tokens': 1024, 'token_safety_margin': 16}
    control_client = OpenAIResponsesBudgetedClient(control_transport, ExperimentBudget(**control_limits))
    control_prompt = official_prompt(control_row)
    admission = control_client.count_messages([{'role': 'user', 'content': control_prompt}])
    raw = CostCappedClient(control_client, _NoopMonetaryGate(), 'headtail_control:0').complete(
        [{'role': 'user', 'content': control_prompt}], max_tokens=1024)
    control_record = {'model': control_transport.model, 'index': 0, 'arm': 'headtail_control',
        'status': 'returned', 'error': None, 'budget': control_limits,
        'prediction': extract_choice(raw), 'strict_official_prediction': official_strict_extract(raw),
        'measurements': {'full_prompt_admission_input_tokens': admission,
            'prepared_admission_input_tokens': admission, 'whole_prompt_fits_control_budget': True},
        'ledger': control_client.ledger, 'count_audit': control_transport.count_audit,
        'response_audit': control_transport.response_audit, 'count_attempt_audit': control_transport.count_attempt_audit}
    control_replay = replay_headtail(control_record, control_value,
        {'direct_admission_tokens_by_model': {control_transport.model: admission}}, [])
    require(control_replay['status'] == 'passed' and control_replay['generation_entries_replayed'] == 1,
            'synthetic_headtail_single_call_replay_failed')
    print(json.dumps({'self_test': 'passed', 'synthetic_generation_entries_replayed': 6,
                      'synthetic_count_operations_retried': 1,
                      'synthetic_headtail_generation_replayed': 1,
                      'real_api_calls': 0, 'experiment_files_opened': 0}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--formal', type=Path, default=ROOT/'work/longbench_openai_limits_20261102/formal')
    parser.add_argument('--recovery', type=Path, default=ROOT/'work/longbench_openai_limits_20261102/recovery')
    parser.add_argument('--control', type=Path, default=ROOT/'work/longbench_openai_limits_20261102/headtail')
    parser.add_argument('--output', type=Path, default=ROOT/'work/longbench_openai_limits_20261102/trajectory_audit.json')
    parser.add_argument('--private-choices-output', type=Path,
        default=ROOT/'work/longbench_openai_limits_20261102/strict_choices_private.json')
    parser.add_argument('--after-final-scoring', '--after-all-114', dest='after_all_114', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    choices = []
    report = audit(args.formal, recovery=args.recovery, control=args.control, authorized_after_all_114=args.after_all_114,
                   private_choices=choices)
    safe_report(report)
    if args.output.exists() or args.private_choices_output.exists():
        raise ReplayUnavailable('refusing_to_overwrite_existing_trajectory_audit')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    args.private_choices_output.parent.mkdir(parents=True, exist_ok=True)
    args.private_choices_output.write_text(json.dumps({'schema_version': 1,
        'official_parser_provenance': report['official_parser_provenance'],
        'contains_raw_reply_or_gold': False, 'created_after_final_score_gates': True,
        'records': choices}, ensure_ascii=False, indent=2)+'\n')
    args.private_choices_output.chmod(0o600)
    print(json.dumps({'status': report['status'], 'totals': report['totals'],
                      'violation_count': len(report['violations'])}))
    return 1 if report['violations'] else 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ReplayUnavailable, ReplayViolation) as exc:
        print(json.dumps({'audit_gate_or_integrity_error': exc.code}))
        raise SystemExit(2)
