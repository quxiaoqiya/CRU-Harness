"""Budget invariants apply equally to root and recursive model requests."""
from long_context_agent.experiment_budget import BudgetedClient, ExperimentBudget, BudgetExhausted, BudgetIntegrityError


class FakeTransport:
    model = 'fake'
    def __init__(self, *, fail=False, actual_prompt=10):
        self.requests = 0
        self.counts = 0
        self.fail = fail
        self.actual_prompt = actual_prompt
    def count(self, **kwargs):
        self.counts += 1
        return 10
    def request(self, *args):
        self.requests += 1
        if self.fail:
            raise TimeoutError('ambiguous delivery')
        return {'choices': [{'message': {'content': 'ok'}}],
                'usage': {'prompt_tokens': self.actual_prompt, 'completion_tokens': 2}}


def test_all_requests_share_physical_call_cap():
    t = FakeTransport()
    c = BudgetedClient(t, ExperimentBudget(max_calls=2))
    c.complete([{'role': 'user', 'content': 'root'}], max_tokens=5)
    c.complete([{'role': 'user', 'content': 'recursive child'}], max_tokens=5)
    assert not c.can_call([], 1)
    counts_before = t.counts
    try:
        c.complete([], max_tokens=1)
    except BudgetExhausted:
        pass
    else:
        raise AssertionError('third physical request admitted')
    assert t.requests == 2
    assert t.counts == counts_before


def test_uncertain_failure_is_charged_and_never_retried():
    t = FakeTransport(fail=True)
    c = BudgetedClient(t, ExperimentBudget(total_tokens=31, max_calls=8))
    try:
        c.complete([], max_tokens=5)
    except TimeoutError:
        pass
    assert t.requests == 1
    assert c.remaining_tokens == 0
    assert c.summary()['charged_tokens'] == 31
    assert not c.can_call([], 1)


def test_context_includes_reserved_output_and_chat_template():
    c = BudgetedClient(FakeTransport(), ExperimentBudget(context_tokens=30))
    assert c.can_call([], 4)
    assert not c.can_call([], 5)


def test_provider_token_disagreement_fails_integrity():
    c = BudgetedClient(FakeTransport(actual_prompt=99), ExperimentBudget())
    try:
        c.complete([], max_tokens=5)
    except BudgetIntegrityError:
        pass
    else:
        raise AssertionError('usage disagreement silently accepted')
    assert c.ledger[0]['status'] == 'error'


def test_live_notice_merges_existing_system_without_mutating_caller():
    c = BudgetedClient(FakeTransport(), ExperimentBudget())
    original = [{'role': 'system', 'content': 'task instruction'}, {'role': 'user', 'content': 'question'}]
    prepared = c._with_budget_notice(original)
    assert sum(m['role'] == 'system' for m in prepared) == 1
    assert 'remaining physical' in prepared[0]['content']
    assert prepared[0]['content'].endswith('task instruction')
    assert original[0]['content'] == 'task instruction'


def test_multiple_leading_systems_are_merged_for_serving_template():
    c = BudgetedClient(FakeTransport(), ExperimentBudget())
    prepared = c._with_budget_notice([{'role': 'system', 'content': 'tools'},
                                      {'role': 'system', 'content': 'persona'},
                                      {'role': 'user', 'content': 'question'}])
    assert [m['role'] for m in prepared] == ['system', 'user']
    assert 'tools\n\npersona' in prepared[0]['content']


def test_native_tools_use_the_same_generation_cap_and_preserve_calls():
    class NativeTransport(FakeTransport):
        def count(self, **kwargs):
            self.tokenize_options = kwargs
            return super().count(**kwargs)
        def request(self, path, body):
            self.requests += 1
            self.body = body
            return {'choices': [{'finish_reason': 'tool_calls', 'message': {
                'role': 'assistant', 'content': None,
                'tool_calls': [{'id': 'call-1', 'type': 'function', 'function': {'name': 'finish', 'arguments': '{}'}}]}}],
                'usage': {'prompt_tokens': 10, 'completion_tokens': 2}}
    t = NativeTransport()
    c = BudgetedClient(t, ExperimentBudget(max_calls=1))
    tools = [{'type': 'function', 'function': {'name': 'finish', 'parameters': {'type': 'object'}}}]
    result = c.complete_native([], tools, max_tokens=8)
    assert result['tool_calls'][0]['function']['name'] == 'finish'
    assert t.tokenize_options['tools'] == t.body['tools'] == tools
    assert t.tokenize_options['tool_choice'] == t.body['tool_choice'] == 'auto'
    assert c.ledger[0]['native_tools'] and c.ledger[0]['tools_sha256']
    assert c.ledger[0]['finish_reason'] == 'tool_calls'
    assert c.remaining_calls == 0


def test_tokenizer_cache_identity_includes_native_schemas():
    from long_context_agent.experiment_budget import ProviderTransport
    t = ProviderTransport(base_url='http://unused.invalid/v1', api_key='fake', model='fake')
    seen = []
    def request(path, body, **kwargs):
        seen.append(body)
        return {'count': len(seen)}
    t.request = request
    a, b = [{'function': {'name': 'a'}}], [{'function': {'name': 'b'}}]
    assert t.count(messages=[], tools=a, tool_choice='auto') == 1
    assert t.count(messages=[], tools=a, tool_choice='auto') == 1
    assert t.count(messages=[], tools=b, tool_choice='auto') == 2
    assert len(seen) == 2
