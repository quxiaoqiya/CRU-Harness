"""Opt-in visual memory with explicit evidence handoff and safe UI events.

V1/V2 and the default text controller are unchanged. Reviews are hypotheses,
not new observations; final packets carry their gaps and omission receipts.
"""
from __future__ import annotations

import ast
from collections import Counter
from dataclasses import asdict
import json
import re

from .adaptive_evidence import _terms, parse_review
from .trajectory_memory import text_sha
from .trajectory_memory_completion import CompletionEvidenceClient, run_completion_harness
from .trajectory_memory_v2 import VisualConfig, VisualStateIndex
from .visual_anchor_memory import AnchorConfig, normalize_perception, PERCEPTION


_CONTROL = {'click', 'dblclick', 'fill', 'clear', 'press', 'keyboard_press',
            'select_option', 'hover', 'focus'}
_EVENTS = _CONTROL | {'go_back', 'go_forward', 'goto', 'scroll', 'tab_focus',
                       'new_tab', 'noop'}


def normalize_actions(raw, before_text):
    """Parse literal UI calls without evaluating code or treating thought as action.

    Multiple calls stay ordered. A malformed/unknown call invalidates the whole
    sequence: keeping only its suffix would invent a partial execution receipt.
    """
    raw = str(raw or '')
    tagged = re.findall(r'<action>\s*(.*?)\s*</action>', raw, re.S | re.I)
    if tagged:
        source = '\n'.join(tagged)
    elif re.search(r'</?(?:think|thought|analysis|action)\b', raw, re.I):
        return []
    else:
        source = raw
    if len(source) > 8192:
        return []
    try:
        nodes = ast.parse(source).body
        if not 1 <= len(nodes) <= 16:
            return []
        result = []
        for node in nodes:
            call = node.value if isinstance(node, ast.Expr) else None
            if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
                return []
            name = call.func.id
            if name not in _EVENTS or any(k.arg is None for k in call.keywords):
                return []
            args = [ast.literal_eval(x) for x in call.args]
            kwargs = {k.arg: ast.literal_eval(k.value) for k in call.keywords}
            # JSON primitives only, no arbitrary AST, objects or huge containers.
            if len(json.dumps([args, kwargs])) > 4096:
                return []
            control = ''
            bid = args[0] if args else kwargs.get('bid')
            if name in _CONTROL and isinstance(bid, (str, int)):
                match = re.search(r'^.*\[' + re.escape(str(bid)) + r'\].*$', before_text, re.M)
                if match:
                    control = match.group().strip()
            result.append({'name': name, 'args': args, 'kwargs': kwargs,
                           'control': control})
        return result
    except (SyntaxError, ValueError, TypeError, RecursionError):
        return []


def normalize_review(raw, valid_ids):
    """Separate required unresolved facts from optional search suggestions."""
    data = parse_review(raw)
    valid = set(valid_ids)
    def strings(value, n=6):
        return [x.strip()[:400] for x in value[:n] if isinstance(x, str) and x.strip()] if isinstance(value, list) else []
    selected = list(dict.fromkeys(x for x in strings(data.get('relevant_ids'), 12) if x in valid))
    visual = list(dict.fromkeys(x for x in strings(data.get('visual_ids'), 6) if x in selected))
    requirements = []
    items = data.get('requirements')
    for item in (items[:8] if isinstance(items, list) else []):
        if not isinstance(item, dict):
            continue
        verdict = item.get('verdict')
        if verdict not in {'support', 'oppose', 'missing'}:
            verdict = 'missing'
        citations = [x for x in strings(item.get('evidence_ids')) if x in valid]
        if verdict in {'support', 'oppose'} and not citations:
            verdict = 'missing'
        requirements.append({'fact': str(item.get('fact', ''))[:300],
                             'verdict': verdict, 'evidence_ids': citations})
    # Old "missing" means a required gap, never an optional search in V3.
    gaps = strings(data.get('required_gaps')) + strings(data.get('missing'), 3)
    gaps += [r['fact'] for r in requirements if r['verdict'] == 'missing' and r['fact']]
    gaps = list(dict.fromkeys(gaps))[:8]
    searches = strings(data.get('followup_queries'), 3)
    if gaps and not searches:
        searches = gaps[:3]
    conflicts = strings(data.get('conflicts'))
    if conflicts and not searches:
        searches = conflicts[:3]
    parse_error = bool(data.get('parse_error'))
    return {'relevant_ids': selected, 'visual_ids': visual,
            'requirements': requirements, 'required_gaps': gaps,
            'followup_queries': searches, 'conflicts': conflicts,
            'complete': data.get('complete') is True and not gaps and not conflicts and not parse_error,
            'reported_complete': data.get('complete') is True,
            'parse_error': parse_error}


def prioritize_evidence(selected, previous, fresh, links, capacity):
    """Promote the latest valid selections; keep observed endpoints together.

    Explicit omissions remain visible when more selected evidence exists than
    the packet can hold. A prior full pin set can never starve all new cards.
    """
    ordered = []
    groups = []
    for key in list(selected) + list(previous) + list(fresh):
        if key in ordered:
            continue
        group = [key]
        for pair in links:
            if key in (pair['before'], pair['after']):
                group = [key] + [x for x in (pair['before'], pair['after']) if x != key]
                break
        if len(ordered) + len([x for x in group if x not in ordered]) <= capacity:
            groups.append(group)
            ordered.extend(x for x in group if x not in ordered)
    return ordered


def needs_transition(question):
    # Only explicit outcome/destination/failed-action relationships. Mere "how"
    # is intentionally insufficient: V2's broad workflow forcing regressed.
    return bool(re.search(r'\b(?:after|before|overwrit\w*|nothing happens|no effect|'
                          r'still\s+\w+|where\s+(?:does|will).{0,60}(?:go|take)|'
                          r'why.{0,80}(?:click|sav|updat|clear|delet)\w*)\b', question, re.I))


class VisualAnchorIndex(VisualStateIndex):
    def __init__(self, store, config=None):
        self.anchor_config = config or AnchorConfig()
        super().__init__(store, VisualConfig(excerpt_chars=self.anchor_config.excerpt_chars))
        self.transitions = []
        self.events_by_state = {}
        for after, (binding, _, state, tree, _, _) in enumerate(self.states):
            before = self.keys.get((binding.trajectory_id, binding.state_index - 1))
            events = normalize_actions(state.get('action'), self.states[before][3] if before is not None else '')
            self.events_by_state[after] = events
            if before is None or not events or all(e['name'] == 'noop' for e in events):
                continue
            control = '\n'.join(e['control'] for e in events if e['control'])
            route = '\n'.join([control, self.states[before][3].split('\n')[0],
                               tree.split('\n')[0], str(state.get('url', '')),
                               json.dumps(events, ensure_ascii=False)])
            self.transitions.append({'before': before, 'after': after,
                'control': control, 'action': json.dumps(events, ensure_ascii=False),
                'terms': _terms(route)})

    def retrieve_anchors(self, queries, perception, seen=()):
        queries = list(dict.fromkeys(q.strip() for q in queries if isinstance(q, str) and q.strip()))
        ranks = [self.rank(q) for q in queries]
        scores = Counter()
        for rank in ranks:
            for pos, number in enumerate(rank[:80]):
                scores[number] += 1 / (60 + pos)
        words = _terms(' '.join(queries))
        ranked_links = []
        for t in self.transitions:
            score = sum(1 / (1 + len(self.postings.get(w, ()))) ** .35 for w in words & t['terms'])
            score += sum(2 for label in perception['visible_labels'] if label.lower() in t['control'].lower())
            score += 10 * (scores[t['before']] + scores[t['after']])
            if score:
                ranked_links.append((score, t))
        ranked_links.sort(key=lambda x: (-x[0], x[1]['after']))
        order, links = [], []
        if perception['transition_needed']:
            for _, t in ranked_links[:2]:
                for number in (t['before'], t['after']):
                    if number not in order:
                        order.append(number)
                links.append(t)
        # Preserve one leading hit per independent search before fusion. A rare
        # exact module/option must not disappear under many generic-page votes.
        for rank in ranks:
            for number in rank:
                if number not in seen:
                    if number not in order:
                        order.append(number)
                    break
        order.extend(n for n, _ in scores.most_common() if n not in order)
        result = []
        for number in order:
            if number in seen:
                continue
            result.append((number, self.card(number, ' '.join(queries))))
            if len(result) == self.anchor_config.max_states:
                break
        # Add receipts only for real incoming events; no causal inference from
        # equal pages, a goal or a successful-looking narrative.
        numbers = {n for n, _ in result}
        for _, t in ranked_links:
            if t['before'] in numbers and t['after'] in numbers and t not in links:
                links.append(t)
        return result, links


REVIEW = (
    'Inspect only the supplied observations. Return JSON: '
    '{"relevant_ids":["actual evidence IDs ordered by usefulness now"],'
    '"visual_ids":["selected IDs whose original pixels are necessary"],'
    '"requirements":[{"fact":"a requested fact or workflow precondition",'
    '"verdict":"support|oppose|missing","evidence_ids":[]}],'
    '"required_gaps":["facts necessary to answer but still unobserved"],'
    '"followup_queries":["up to three precise new history searches"],'
    '"conflicts":["unresolved disagreements or entity/role mismatches"],'
    '"complete":false}. Optional corroboration searches do not constitute required '
    'gaps. Set complete only if required gaps are empty. A current entity and a '
    'historical example are distinct unless observed identity matches. For effects '
    'seek before/action/after or settled saved state; the presence of a control '
    'does not prove its effect. Do not answer yet. No confidence threshold.'
)

FINAL_POLICY = (
    '\nUse original current and historical observations. Historical records are '
    'examples of this deployment, not automatically the current user or object. '
    'Keep identity, role, field values and action preconditions separate. Generated '
    'routing and review notes are hypotheses, never independent evidence. Normalized '
    'UI events record actions, not successful effects. Check observed after states '
    'before inferring a workflow. The review handoff lists unresolved gaps, conflicts '
    'and evidence omitted by budget; do not silently treat these as resolved. A '
    'gap may be resolved from the supplied observations; an optional search is not '
    'a reason to refuse. Correct contradicted premises explicitly. If a necessary '
    'fact remains unsupported, say I do not know rather than invent it. Preserve '
    'exact UI labels and put the final answer in \\boxed{...}.'
)


def run_visual_anchor(index, question, client, *, domain_system, query_image, query_only=False):
    store, config = index.store, index.anchor_config
    if not query_image:
        return run_completion_harness(store, question, CompletionEvidenceClient(
            client, store, domain_system=domain_system, question_text=question))
    url, query_sha = store.image_payload(query_image)
    query_receipt = {'query_image': True, 'image_sha256': query_sha}
    events, trace = [], []
    def call(messages, output, images, evidence=(), transitions=()):
        if not client.can_call(messages, output):
            return None
        reply = client.complete(messages, output)
        events.append({'physical_request': len(client.ledger),
            'prepared_messages_sha256': text_sha(json.dumps(messages, sort_keys=True, ensure_ascii=False)),
            'images': images, 'evidence_receipts': [c.receipt() for c in evidence],
            'transition_receipts': transitions})
        return reply
    raw = call([{'role': 'system', 'content': PERCEPTION}, {'role': 'user', 'content': [
        {'type': 'text', 'text': question}, {'type': 'image_url', 'image_url': {'url': url}}]}],
        config.perception_output, [query_receipt])
    perception = normalize_perception(raw or '')
    perception['transition_needed'] |= needs_transition(question)
    queries = perception['retrieval_queries'] + [question, ' '.join(perception['visible_labels'])]
    queries += perception['visible_state_facts'][:3]
    bank, seen, priority, all_links = {}, set(), [], []
    latest = normalize_review('{}', [])
    selected_history, visual_priority = [], []

    def assemble(cards, task):
        ids = {c.id for c in cards}
        by_state = {(c.trajectory_id, c.state_index): c.id for c in cards}
        receipts = []
        for t in all_links:
            a, b = index.states[t['before']][0], index.states[t['after']][0]
            before, after = by_state.get((a.trajectory_id, a.state_index)), by_state.get((b.trajectory_id, b.state_index))
            if before and after:
                pair = {'before': before, 'after': after, 'incoming_action': t['action'], 'observed_control': t['control']}
                if pair not in receipts:
                    receipts.append(pair)
        omitted = [x for x in latest['relevant_ids'] if x not in ids]
        handoff = {**latest, 'selected_evidence_omitted': omitted,
                   'review_notes_are_not_observations': True}
        rendered = []
        for card in cards:
            n = index.keys[card.trajectory_id, card.state_index]
            rendered.append(json.dumps({'evidence_id': card.id, 'trajectory': card.trajectory_id,
                'state': card.state_index, 'observed_url': card.url,
                'identity_scope': 'historical example; compare identity with current screenshot',
                'normalized_incoming_events': index.events_by_state.get(n, []),
                'observed_field': card.field, 'complete_field': card.complete_field}, ensure_ascii=False)
                + '\n' + '\n[...source field gap...]\n'.join(card.excerpts))
        parts = [{'type': 'text', 'text': task + '\nGenerated routing hints:\n'
            + json.dumps(perception, ensure_ascii=False) + '\nReview handoff (hypotheses):\n'
            + json.dumps(handoff, ensure_ascii=False) + '\nObserved transition pairs:\n'
            + json.dumps(receipts, ensure_ascii=False) + '\nObserved history fields:\n' + '\n\n'.join(rendered)}]
        image_order = list(dict.fromkeys(visual_priority + [c.id for c in cards]))
        images = []
        card_map = {c.id: c for c in cards}
        for key in image_order:
            card = card_map.get(key)
            if not card or not card.image or len(images) >= config.history_images:
                continue
            payload, sha = store.image_payload(card.image)
            if any(x['image_sha256'] == sha for x in images):
                continue
            aliases = [c.id for c in cards if c.image and store.image_payload(c.image)[1] == sha]
            parts += [{'type': 'text', 'text': 'Original history image for ' + ', '.join(aliases)},
                      {'type': 'image_url', 'image_url': {'url': payload}}]
            images.append({**card.receipt(), 'image_sha256': sha, 'alias_evidence_ids': aliases})
        exposed = {x for img in images for x in img['alias_evidence_ids']}
        parts += [{'type': 'text', 'text': 'History evidence IDs with original pixels in THIS packet: '
                   + json.dumps(sorted(exposed)) + '\nHistory evidence IDs with text only in THIS packet: '
                   + json.dumps(sorted(ids - exposed))},
                  {'type': 'text', 'text': 'Current question and original current screenshot:\n' + question},
                  {'type': 'image_url', 'image_url': {'url': url}}]
        images.append(query_receipt)
        return [{'role': 'system', 'content': domain_system + FINAL_POLICY},
                {'role': 'user', 'content': parts}], images, receipts

    if not query_only:
        for round_index in range(config.max_rounds):
            if client.remaining_calls < 2 or client.remaining_tokens < 8192:
                break
            fresh, links = index.retrieve_anchors(queries, perception, seen)
            if not fresh:
                break
            for number, card in fresh:
                bank[card.id] = card
            all_links.extend(t for t in links if t not in all_links)
            # Fresh-first review prevents the previous packet from consuming the
            # entire admission budget before the follow-up is ever examined.
            review_order = [c.id for _, c in fresh] + priority
            cards = [bank[x] for x in dict.fromkeys(review_order) if x in bank]
            while cards:
                messages, images, pairs = assemble(cards, REVIEW)
                if client.can_call(messages, config.review_output) and client.remaining_tokens >= client.count_messages(messages) + config.review_output + 8192:
                    break
                cards.pop()
            if not cards:
                break
            latest = normalize_review(call(messages, config.review_output, images, cards, pairs) or '', [c.id for c in cards])
            seen.update(n for n, c in fresh if c.id in {x.id for x in cards})
            selected = latest['relevant_ids']
            selected_history.append(selected)
            visual_priority = latest['visual_ids'] + [x for x in visual_priority if x not in latest['visual_ids']]
            priority = prioritize_evidence(selected, priority, [c.id for _, c in fresh], pairs, config.max_states)
            queries = latest['followup_queries']
            trace.append({'round': round_index, 'card_ids': [c.id for c in cards],
                'fresh_ids': [c.id for _, c in fresh], 'selected_ids': selected,
                'priority_ids': priority[:], 'review': latest,
                'followup_queries_sha256': text_sha(json.dumps(queries)),
                'complete': latest['complete']})
            if latest['complete'] or not queries:
                break
    cards = [bank[x] for x in priority if x in bank][:config.max_states]
    answer, final_sent = '', False
    for output in sorted({config.final_output, 2048, 1024}, reverse=True):
        final, images, pairs = assemble(cards, 'Answer the user question from original current and historical evidence.')
        if client.can_call(final, output):
            answer = call(final, output, images, cards, pairs) or ''
            final_sent = True
            break
    # Latest selected evidence is first, so shrinking removes stale tail first.
    while not final_sent and cards:
        cards.pop()
        final, images, pairs = assemble(cards, 'Answer the user question from original current and historical evidence.')
        if client.can_call(final, 1024):
            answer = call(final, 1024, images, cards, pairs) or ''
            final_sent = True
    final_ids = [c.id for c in cards]
    final_images = events[-1]['images'] if final_sent else []
    exposed = {x for img in final_images for x in img.get('alias_evidence_ids', [])}
    return {'mode': 'visual_query_perception_only_v3' if query_only else 'visual_anchor_memory_v3',
        'answer': answer, 'source_sha256': text_sha(store.context), 'config': asdict(config),
        'perception': perception, 'trace': trace, 'image_events': events,
        'final_evidence_ids': final_ids, 'pinned_evidence_ids': latest['relevant_ids'],
        'pinned_omitted': [x for x in latest['relevant_ids'] if x not in final_ids],
        'latest_review': latest, 'final_selected_retained': [x for x in latest['relevant_ids'] if x in final_ids],
        'final_visual_omitted': [x for x in latest['visual_ids'] if x not in exposed],
        'no_confidence_gate': True}
