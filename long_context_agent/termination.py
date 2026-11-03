class Termination:
    def __init__(self, max_actions=256): self.max_actions=max_actions
    def should_stop(self, state, memory, *, valid=False): return (valid and not memory.conflicts.unresolved() and not memory.missing_fields) or state.action_index >= self.max_actions
