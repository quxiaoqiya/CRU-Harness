"""Opt-in multimodal trajectory memory; existing text controllers are unchanged.

Raw trajectories remain outside the reader. Retrieval reuses SourceIndex and
the evidence-gap protocol, while screenshots retain state/span/hash bindings.
No benchmark answer, question ID, type or scoring function enters this module.
"""
from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
import base64
import copy
import hashlib
import json
import mimetypes
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

from .adaptive_evidence import (EvidenceGap, EvidenceWindow, SourceIndex,
    _allocate, _gaps, _messages, parse_review, source_coverage,
    validate_supported_spans)


def text_sha(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


@dataclass(frozen=True)
class StateBinding:
    trajectory_id: str
    state_index: int
    start: int
    end: int
    image: str | None


@dataclass(frozen=True)
class TrajectoryConfig:
    block_chars: int = 1600
    halo_chars: int = 240
    max_windows_per_round: int = 8
    max_rounds: int = 3
    review_output_tokens: int = 1024
    answer_output_tokens: int = 1024
    max_history_images_per_call: int = 3


class TrajectoryStore:
    """Lossless text projection with addressable original screenshot assets."""
    def __init__(self, data_root: str | Path, *, config: TrajectoryConfig | None = None):
        self.data_root = Path(data_root).resolve()
        self.config = config or TrajectoryConfig()
        self._rows: dict[str, dict[str, Any]] = {}
        self.context = ''
        self.bindings: list[StateBinding] = []
        self._ends: list[int] = []
        self.index: SourceIndex | None = None
        self._image_cache: dict[str, tuple[str, str]] = {}
        self._metadata_cache: dict[str, Any] | None = None

    def insert(self, trajectory: Mapping[str, Any]) -> None:
        forbidden = {'answer', 'gold', 'eval_function', 'question_type', 'question_id'}
        if forbidden & set(trajectory):
            raise ValueError('Benchmark labels or query metadata cannot enter trajectory memory')
        tid = trajectory.get('id')
        if not isinstance(tid, str) or not tid or not isinstance(trajectory.get('states'), list):
            raise ValueError('Trajectory needs an ID and a state list')
        row = json.loads(json.dumps(dict(trajectory), ensure_ascii=False))
        if tid in self._rows and self._rows[tid] != row:
            raise ValueError('A trajectory ID cannot silently change its source')
        self._rows[tid] = row
        self.index = None
        self._metadata_cache = None

    def finalize(self) -> None:
        if self.index is not None:
            return
        chunks, bindings, cursor = [], [], 0
        for tid, row in self._rows.items():
            header = '\n[trajectory '+json.dumps({key: row.get(key) for key in
                ('id', 'domain', 'environment', 'goal', 'outcome', 'start_url')}, ensure_ascii=False)+']\n'
            chunks.append(header);cursor += len(header)
            for number, state in enumerate(row['states']):
                if not isinstance(state, dict):
                    raise ValueError('Trajectory state must be an object')
                # Preserve every original state field in canonical JSON. No
                # goal/action/tree/observation is replaced by a model summary.
                body = '[state '+str(number)+']\n'+json.dumps(state, ensure_ascii=False, sort_keys=True)+'\n'
                image = state.get('screenshot')
                if image is not None and (not isinstance(image, str) or not image):
                    raise ValueError('Screenshot must be an asset path or null')
                bindings.append(StateBinding(tid, number, cursor, cursor+len(body), image))
                chunks.append(body);cursor += len(body)
        self.context = ''.join(chunks)
        self.bindings = bindings
        self._ends = [item.end for item in bindings]
        self.index = SourceIndex(self.context, block_chars=self.config.block_chars,
                                 halo_chars=self.config.halo_chars)

    def resolve_image(self, relative: str) -> Path:
        path = (self.data_root/relative).resolve()
        if not path.is_relative_to(self.data_root) or not path.is_file():
            raise ValueError('Missing screenshot or screenshot outside the data root')
        return path

    def image_payload(self, relative: str) -> tuple[str, str]:
        if relative not in self._image_cache:
            path = self.resolve_image(relative)
            raw = path.read_bytes()
            mime = mimetypes.guess_type(path.name)[0] or 'image/png'
            self._image_cache[relative] = ('data:'+mime+';base64,'+base64.b64encode(raw).decode('ascii'),
                                          hashlib.sha256(raw).hexdigest())
        return self._image_cache[relative]

    def images_for_spans(self, spans: Iterable[tuple[int, int]], *, limit: int,
                         selection: str = 'overlap') -> list[StateBinding]:
        self.finalize()
        scores: dict[StateBinding, int] = {}
        for start, end in spans:
            if start < 0 or end > len(self.context) or end <= start:
                raise ValueError('Image lookup requires a valid source span')
            i = bisect_left(self._ends, start+1)
            while i < len(self.bindings) and self.bindings[i].start < end:
                state = self.bindings[i]
                if state.image:
                    overlap = max(0, min(end, state.end)-max(start, state.start))
                    scores[state] = scores.get(state, 0)+overlap
                i += 1
        ranked = sorted(scores, key=lambda state: (-scores[state], state.start))
        if selection == 'uniform':
            ranked = sorted(ranked, key=lambda state: state.start)
            # A chronology-only control gets broad image coverage, without
            # using the question or oracle evidence to select screenshots.
            unique = list(dict.fromkeys(state.image for state in ranked))
            first = {state.image: state for state in reversed(ranked)}
            ranked = [first[name] for name in unique]
            if limit > 1 and len(ranked) > limit:
                ranked = [ranked[round(i*(len(ranked)-1)/(limit-1))] for i in range(limit)]
        elif selection != 'overlap':
            raise ValueError('Unknown image selection policy')
        chosen, seen = [], set()
        for state in ranked:
            if state.image in seen:
                continue
            seen.add(state.image);chosen.append(state)
            if len(chosen) >= max(0, limit):
                break
        return chosen if limit > 0 else []

    def metadata(self) -> dict[str, Any]:
        self.finalize()
        if self._metadata_cache is not None:
            return copy.deepcopy(self._metadata_cache)
        self._metadata_cache = {'trajectory_count': len(self._rows), 'state_count': len(self.bindings),
            'source_chars': len(self.context), 'canonical_source_sha256': text_sha(self.context),
            'source_trajectory_sha256': {tid: text_sha(json.dumps(row, sort_keys=True, ensure_ascii=False))
                                         for tid, row in self._rows.items()},
            'history_screenshot_references': sum(bool(s.image) for s in self.bindings)}
        return copy.deepcopy(self._metadata_cache)


_HEADER = re.compile(r'^\[source_chars (\d+):(\d+) \| gap ', re.M)


class TrajectoryEvidenceClient:
    """Attach images deterministically before BOTH admission and generation."""
    def __init__(self, client: Any, store: TrajectoryStore, *, domain_system: str,
                 query_image: str | None = None, history_image_limit: int | None = None,
                 image_selection: str = 'overlap', question_text: str | None = None):
        self.client, self.store = client, store
        self.domain_system, self.query_image = domain_system, query_image
        self.history_image_limit = (store.config.max_history_images_per_call
                                    if history_image_limit is None else int(history_image_limit))
        self.image_selection = image_selection
        self.question_text = question_text
        self.image_events: list[dict[str, Any]] = []

    @property
    def context_tokens(self): return self.client.context_tokens
    @property
    def remaining_tokens(self): return self.client.remaining_tokens
    @property
    def remaining_calls(self): return self.client.remaining_calls

    def prepare(self, messages):
        prepared = [dict(message) for message in messages]
        is_review = 'evidence-gap reviewer' in str(prepared[0].get('content', ''))
        official_final = not is_review and self.question_text is not None
        text = '\n'.join(str(m.get('content', '')) for m in messages)
        spans = list(dict.fromkeys((int(a), int(b)) for a, b in _HEADER.findall(text)))
        images = self.store.images_for_spans(spans, limit=self.history_image_limit,
                                           selection=self.image_selection) if spans else []
        evidence = []
        raw = str(prepared[-1]['content'])
        if official_final:
            evidence_text = raw.partition('\n\nRaw source evidence:\n')[2]
            parts = [{'type': 'text', 'text': '### Memory context:\n'+('' if evidence_text else '(empty)')}]
            if evidence_text:
                parts.append({'type': 'text', 'text': evidence_text})
        else:
            parts = [{'type': 'text', 'text': raw}]
        for state in images:
            payload, digest = self.store.image_payload(state.image)
            label = f'Screenshot evidence: trajectory {state.trajectory_id}, state {state.state_index}; source state chars {state.start}:{state.end}.'
            parts.extend([{'type': 'text', 'text': label}, {'type': 'image_url', 'image_url': {'url': payload}}])
            evidence.append({'trajectory_id': state.trajectory_id, 'state_index': state.state_index,
                'source_state_span': [state.start, state.end], 'image_sha256': digest})
        if official_final:
            parts.append({'type': 'text', 'text': '\n\n### Question to answer:\n'+self.question_text})
        if self.query_image:
            payload, digest = self.store.image_payload(self.query_image)
            if not official_final:
                parts.append({'type': 'text', 'text': 'Screenshot supplied with the question:'})
            parts.append({'type': 'image_url', 'image_url': {'url': payload}})
            evidence.append({'query_image': True, 'image_sha256': digest})
        prepared[-1]['content'] = parts
        suffix = (' For this intermediate evidence review, keep the required JSON schema. '
                  'An UNKNOWN answer is allowed; a flawed premise needs an explicit explanation.' if is_review else
                  ' Return the final answer in \\boxed{...}. Use \\boxed{UNKNOWN} when evidence is insufficient; do not guess.')
        prepared[0]['content'] = (self.domain_system if official_final else
            str(prepared[0].get('content', ''))+'\n\nTask policy: '+self.domain_system+suffix)
        return prepared, evidence

    def count_text(self, text): return self.client.count_text(text)
    def count_messages(self, messages): return self.client.count_messages(self.prepare(messages)[0])
    def can_call(self, messages, max_tokens=1024): return self.client.can_call(self.prepare(messages)[0], max_tokens)
    def complete(self, messages, max_tokens=1024):
        prepared, evidence = self.prepare(messages)
        before = len(self.client.ledger)
        try:
            return self.client.complete(prepared, max_tokens=max_tokens)
        finally:
            if len(self.client.ledger) > before:
                self.image_events.append({'physical_request': len(self.client.ledger), 'images': evidence,
                    'prepared_messages_sha256': text_sha(json.dumps(prepared, sort_keys=True, ensure_ascii=False))})


def retrieve_trajectory_evidence(store: TrajectoryStore, question: str, client: TrajectoryEvidenceClient) -> dict[str, Any]:
    """Evidence gaps, novel windows and pinned support share one bounded client."""
    store.finalize()
    config = store.config
    gaps = [EvidenceGap('question', question, 'Find observed UI states and workflows supporting the answer')]
    bank: dict[tuple[int, int], EvidenceWindow] = {}
    seen, supported = set(), set()
    trace, tentative, stop = [], '', 'round_limit'
    for round_index in range(config.max_rounds):
        if client.remaining_calls <= 1 or client.remaining_tokens < 4096:
            stop = 'final_call_reserved';break
        output = config.review_output_tokens
        overhead = client.count_messages(_messages(question, [], gaps=gaps, tentative=tentative))
        allocation = min(7000, client.context_tokens-overhead-output-2048,
                         (client.remaining_tokens-4096)//3)
        if allocation <= 128:
            stop = 'image_or_text_budget_reserved';break
        fresh, allocations = _allocate(store.index, gaps, seen, client, allocation, config.max_windows_per_round)
        if not fresh:
            stop = 'no_novel_evidence';break
        exposed = list(fresh)
        pinned = [window for window in bank.values() if window.span in supported]
        for old in pinned:
            if client.can_call(_messages(question, exposed+[old], gaps=gaps, tentative=tentative), output):
                exposed.append(old)
        while exposed and not client.can_call(_messages(question, exposed, gaps=gaps, tentative=tentative), output):
            exposed.pop()
        if not exposed:
            stop = 'multimodal_request_does_not_fit';break
        novel = [window for window in exposed if (window.start, window.end) not in seen]
        if not novel:
            stop = 'no_novel_evidence';break
        for window in novel:
            seen.add((window.start, window.end));bank[window.start, window.end] = window
        review = parse_review(client.complete(_messages(question, exposed, gaps=gaps, tentative=tentative), output))
        good, bad = validate_supported_spans(review.get('supported_spans'), exposed)
        supported.update(good)
        if isinstance(review.get('answer'), str) and review['answer'].strip():
            tentative = review['answer'].strip()
        next_gaps = _gaps(review.get('needs'), [])
        complete = review.get('complete') is True and bool(good) and not bad and not next_gaps and bool(tentative)
        trace.append({'round': round_index, 'allocation': allocations,
            'new_spans': [w.span for w in novel], 'exposed_spans': [w.span for w in exposed],
            'supported_spans': good, 'invalid_spans': bad,
            'parse_error': bool(review.get('parse_error')), 'complete': complete,
            'next_gaps': [gap.as_dict() for gap in next_gaps]})
        if complete:
            gaps = [];stop = 'evidence_complete';break
        gaps = next_gaps or [EvidenceGap('followup', question, 'Seek new evidence or conflicts; repeated text does not resolve a gap')]
    windows = sorted(bank.values(), key=lambda w: (w.span not in supported, -w.score, w.start))
    final = _messages(question, windows, gaps=gaps, tentative=tentative, final=True)
    while windows and not client.can_call(final, config.answer_output_tokens):
        windows.pop();final = _messages(question, windows, gaps=gaps, tentative=tentative, final=True)
    answer = ''
    if client.can_call(final, config.answer_output_tokens):
        answer = client.complete(final, config.answer_output_tokens)
    return {'mode': 'trajectory_evidence_harness', 'answer': answer, 'trace': trace,
        'stop_reason': stop, 'source_sha256': text_sha(store.context),
        'source_coverage': source_coverage(seen, len(store.context)),
        'supported_spans': sorted(supported), 'final_spans': [w.span for w in windows],
        'supported_spans_not_in_final': sorted(supported-{w.span for w in windows}),
        'image_events': client.image_events}
