"""Opt-in screenshot-grounded recall with observed control transitions.

Query perception routes retrieval; original images and observed source fields
remain the evidence. No benchmark IDs, labels or answer templates enter here.
"""
from __future__ import annotations

import ast
from collections import Counter
from dataclasses import asdict, dataclass
import json
import re

from .adaptive_evidence import _terms, parse_review
from .trajectory_memory import text_sha
from .trajectory_memory_completion import (
    CompletionEvidenceClient, run_completion_harness,
)
from .trajectory_memory_v2 import VisualConfig, VisualStateIndex


@dataclass(frozen=True)
class AnchorConfig:
    perception_output: int = 1536
    review_output: int = 1536
    final_output: int = 4096
    max_rounds: int = 2
    max_states: int = 6
    history_images: int = 3
    excerpt_chars: int = 2600


def action_target(action, before_text):
    """Locate the observed control for an incoming action; never execute it."""
    try:
        node = ast.parse(str(action)).body[0]
        call = node.value if isinstance(node, ast.Expr) else None
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
            return None
        name = call.func.id
        if name not in {'click', 'fill', 'select_option', 'keyboard_press',
                        'go_back', 'go_forward', 'goto', 'scroll', 'hover'}:
            return None
        target = ''
        if name in {'click', 'fill', 'select_option', 'hover'} and call.args:
            bid = ast.literal_eval(call.args[0])
            if isinstance(bid, (str, int)):
                match = re.search(r'^.*\[' + re.escape(str(bid)) + r'\].*$',
                                  before_text, re.M)
                if match:
                    target = match.group().strip()
        return name, target
    except (ValueError, SyntaxError, TypeError, IndexError):
        return None


def normalize_perception(raw):
    data = parse_review(raw)
    def strings(key, limit, chars):
        value = data.get(key)
        return [x.strip()[:chars] for x in value[:limit]
                if isinstance(x, str) and x.strip()] if isinstance(value, list) else []
    return {
        'visible_labels': strings('visible_labels', 16, 120),
        'visible_state_facts': strings('visible_state_facts', 10, 180),
        'retrieval_queries': strings('retrieval_queries', 4, 320),
        'transition_needed': data.get('transition_needed') is True,
        'parse_error': bool(data.get('parse_error')),
    }


class VisualAnchorIndex(VisualStateIndex):
    def __init__(self, store, config=None):
        self.anchor_config = config or AnchorConfig()
        super().__init__(store, VisualConfig(excerpt_chars=self.anchor_config.excerpt_chars))
        self.transitions = []
        for after, (binding, _, state, tree, _, _) in enumerate(self.states):
            before = self.keys.get((binding.trajectory_id, binding.state_index - 1))
            if before is None:
                continue
            previous = self.states[before]
            target = action_target(state.get('action'), previous[3])
            if target is None:
                continue
            # Dataset action at state i led from i-1 into i, not out of i.
            route = '\n'.join([target[1], previous[3].split('\n')[0],
                               tree.split('\n')[0], str(previous[2].get('url', '')),
                               str(state.get('url', '')), str(state.get('action', ''))])
            self.transitions.append({'before': before, 'after': after,
                'control': target[1], 'action': str(state.get('action', '')),
                'terms': _terms(route)})

    def retrieve_anchors(self, queries, perception, seen=()):
        queries = [q for q in queries if isinstance(q, str) and q.strip()]
        ranks = [self.rank(q) for q in queries]
        state_score = Counter()
        for rank in ranks:
            for position, number in enumerate(rank[:80]):
                state_score[number] += 1 / (60 + position)
        words = _terms(' '.join(queries))
        labels = [x.lower() for x in perception['visible_labels']]
        transitions = []
        for t in self.transitions:
            score = sum(1 / (1 + len(self.postings.get(w, ()))) ** .35
                        for w in words & t['terms'])
            score += sum(2 for label in labels if label in t['control'].lower())
            score += 10 * (state_score[t['before']] + state_score[t['after']])
            if score:
                transitions.append((score, t))
        transitions.sort(key=lambda item: (-item[0], item[1]['after']))
        order = []
        links = []
        # Transition pairs precede isolated matches when an effect/destination is asked.
        if perception['transition_needed']:
            for _, transition in transitions[:2]:
                for number in [transition['before'], transition['after']]:
                    if number not in order:
                        order.append(number)
                links.append(transition)
        for number, _ in state_score.most_common():
            if number not in order:
                order.append(number)
            if len(order) >= self.anchor_config.max_states + len(seen):
                break
        if not perception['transition_needed']:
            for _, transition in transitions[:1]:
                links.append(transition)
        result = []
        for number in order:
            if number in seen:
                continue
            # State identity is retained even if accessibility text is identical.
            result.append((number, self.card(number, ' '.join(queries))))
            if len(result) == self.anchor_config.max_states:
                break
        return result, links


PERCEPTION = (
    'Inspect the original current screenshot. Return JSON only: '
    '{"visible_labels":["exact page, module and control labels"],'
    '"visible_state_facts":["exact visible label: value, including status, role, '
    'selection and disabled/enabled state relevant to the question"],'
    '"retrieval_queries":["up to four saved-history searches for relevant observed '
    'states, failed attempts, successful alternatives, or actual control destinations"],'
    '"transition_needed":false}. Do not answer the user question or recommend an '
    'action. Read actual values before inferring affordances. A current screenshot '
    'does not establish what a future click does. Keep JSON under 600 words.'
)

FINAL_POLICY = (
    '\nUse the current original screenshot to check the current state, role and '
    'action preconditions. Historical states may differ; do not overwrite current '
    'visible values with a similar historical page. Generated perception/search '
    'notes are routing hints, not independent evidence. A visible control proves '
    'its presence, not a successful effect or destination. For workflows use '
    'observed before/action/after pairs. If a premise conflicts with the supplied '
    'observations, explain that correction explicitly. If the evidence cannot '
    'establish the requested fact, use \\boxed{UNKNOWN}. Preserve exact UI labels '
    'and return the final answer in \\boxed{...}.'
)


def run_visual_anchor(index, question, client, *, domain_system, query_image,
                      query_only=False):
    store, config = index.store, index.anchor_config
    if not query_image:
        adapted = CompletionEvidenceClient(client, store, domain_system=domain_system,
                                            question_text=question)
        return run_completion_harness(store, question, adapted)
    url, query_sha = store.image_payload(query_image)
    events = []
    def call(messages, output, images, evidence=(), transitions=()):
        if not client.can_call(messages, output):
            return None
        reply = client.complete(messages, output)
        events.append({'physical_request': len(client.ledger),
            'prepared_messages_sha256': text_sha(json.dumps(messages, sort_keys=True, ensure_ascii=False)),
            'images': images, 'evidence_receipts': [c.receipt() for c in evidence],
            'transition_receipts': transitions})
        return reply
    query_receipt = {'query_image': True, 'image_sha256': query_sha}
    messages = [{'role': 'system', 'content': PERCEPTION}, {'role': 'user', 'content': [
        {'type': 'text', 'text': question}, {'type': 'image_url', 'image_url': {'url': url}}]}]
    raw = call(messages, config.perception_output, [query_receipt])
    perception = normalize_perception(raw or '')
    bank, seen, pinned, trace = {}, set(), [], []
    queries = perception['retrieval_queries'] + [question, ' '.join(perception['visible_labels'])]

    def assemble(cards, task, links=()):
        cards = list(cards)
        by_state = {(c.trajectory_id, c.state_index): c.id for c in cards}
        receipts = []
        for t in links:
            before = index.states[t['before']][0]
            after = index.states[t['after']][0]
            a, b = by_state.get((before.trajectory_id, before.state_index)), by_state.get((after.trajectory_id, after.state_index))
            if a and b:
                receipts.append({'before': a, 'after': b, 'incoming_action': t['action'],
                                 'observed_control': t['control']})
        rendered = []
        for card in cards:
            rendered.append(json.dumps({'evidence_id': card.id, 'trajectory': card.trajectory_id,
                'state': card.state_index, 'observed_url': card.url,
                'action_leading_into_this_state': card.action, 'observed_field': card.field,
                'complete_field': card.complete_field}, ensure_ascii=False)
                + '\n' + '\n[...source field gap...]\n'.join(card.excerpts))
        parts = [{'type': 'text', 'text': task + '\nGenerated routing hints, verify against images:\n'
            + json.dumps(perception, ensure_ascii=False) + '\nObserved transition pairs:\n'
            + json.dumps(receipts, ensure_ascii=False) + '\nObserved history fields:\n'
            + '\n\n'.join(rendered)}]
        images = []
        for card in cards:
            if not card.image or len(images) >= config.history_images:
                continue
            payload, sha = store.image_payload(card.image)
            if any(x['image_sha256'] == sha for x in images):
                continue
            parts.extend([{'type': 'text', 'text': 'Original history image for ' + card.id},
                          {'type': 'image_url', 'image_url': {'url': payload}}])
            images.append({**card.receipt(), 'image_sha256': sha})
        parts.extend([{'type': 'text', 'text': 'Current question and original current screenshot:\n' + question},
                      {'type': 'image_url', 'image_url': {'url': url}}])
        images.append(query_receipt)
        return [{'role': 'system', 'content': domain_system + FINAL_POLICY},
                {'role': 'user', 'content': parts}], images, receipts

    all_links = []
    if not query_only:
        for round_index in range(config.max_rounds):
            if client.remaining_calls < 2 or client.remaining_tokens < 8192:
                break
            fresh, links = index.retrieve_anchors(queries, perception, seen)
            if not fresh:
                break
            for number, card in fresh:
                bank[card.id] = card
            all_links.extend(links)
            cards = [bank[x] for x in pinned if x in bank]
            cards += [card for _, card in fresh if card.id not in pinned]
            task = ('Inspect only the supplied observations. Return JSON with relevant_ids '
                    '(actual evidence IDs), missing (up to three new saved-history searches), '
                    'complete (boolean), and conflicts (observed disagreements). Check current '
                    'state preconditions and actual destinations. Do not answer yet. A page '
                    'goal or common product workflow is not evidence of a successful action.')
            while cards:
                m, images, pairs = assemble(cards, task, all_links)
                if client.can_call(m, config.review_output) and client.remaining_tokens >= client.count_messages(m) + config.review_output + 8192:
                    break
                cards.pop()
            if not cards:
                break
            review = parse_review(call(m, config.review_output, images, cards, pairs) or '')
            seen.update(number for number, card in fresh if card.id in {c.id for c in cards})
            valid = {c.id for c in cards}
            ids = review.get('relevant_ids')
            selected = [x for x in ids if isinstance(x, str) and x in valid] if isinstance(ids, list) else []
            groups = []
            for selected_id in pinned + selected:
                group = [selected_id]
                for pair in pairs:
                    if selected_id in [pair['before'], pair['after']]:
                        group = [pair['before'], pair['after']]
                        break
                if group not in groups:
                    groups.append(group)
            pinned = []
            for group in groups:
                new = [x for x in group if x not in pinned]
                if len(pinned) + len(new) <= 4:
                    pinned.extend(new)
            missing = review.get('missing')
            queries = [x[:400] for x in missing[:3] if isinstance(x, str) and x.strip()] if isinstance(missing, list) else []
            trace.append({'round': round_index, 'card_ids': [c.id for c in cards],
                          'pinned_ids': pinned[:], 'parse_error': bool(review.get('parse_error')),
                          'followup_queries_sha256': text_sha(json.dumps(queries)),
                          'complete': review.get('complete') is True})
            if review.get('complete') is True or not queries:
                break
    cards = [bank[x] for x in pinned if x in bank]
    cards += [card for key, card in bank.items() if key not in pinned]
    cards = cards[:config.max_states]
    answer = ''
    final_sent = False
    for output in sorted({config.final_output, 2048, 1024}, reverse=True):
        final, images, pairs = assemble(cards, 'Answer the user question from original current and historical evidence.', all_links)
        if client.can_call(final, output):
            answer = call(final, output, images, cards, pairs) or ''
            final_sent = True
            break
    while not final_sent and cards and cards[-1].id not in pinned:
        cards.pop()
        final, images, pairs = assemble(cards, 'Answer the user question from original current and historical evidence.', all_links)
        if client.can_call(final, 1024):
            answer = call(final, 1024, images, cards, pairs) or ''
            final_sent = True
            break
    return {'mode': 'visual_query_perception_only' if query_only else 'visual_anchor_memory',
            'answer': answer, 'source_sha256': text_sha(store.context), 'config': asdict(config),
            'perception': perception, 'trace': trace, 'image_events': events,
            'final_evidence_ids': [c.id for c in cards], 'pinned_evidence_ids': pinned,
            'pinned_omitted': [x for x in pinned if x not in {c.id for c in cards}],
            'no_confidence_gate': True}
