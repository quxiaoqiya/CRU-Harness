"""Budget and completion assistance without changing native reasoning order."""
from dataclasses import asdict,dataclass
import copy,json,re
from .experiment_budget import BudgetedClient
from .trajectory_memory import TrajectoryEvidenceClient,retrieve_trajectory_evidence,text_sha
from .trajectory_memory_assist import AssistConfig,AssistEvidenceClient,boxed_answer


class PerCallBudgetedClient(BudgetedClient):
    """Count/generate with the actual request's output notice, not a larger cap."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self._notice_output=min(1024,self.budget.max_output_tokens)
    def _with_budget_notice(self,messages):
        prepared=super()._with_budget_notice(messages)
        old=f'maximum output per request: {self.budget.max_output_tokens} tokens.'
        new=f'maximum output per request: {self._notice_output} tokens.'
        prepared[0]['content']=prepared[0]['content'].replace(old,new,1)
        return prepared
    def can_call(self,messages,max_tokens=2048,**kwargs):
        self._notice_output=min(int(max_tokens),self.budget.max_output_tokens)
        return super().can_call(messages,max_tokens,**kwargs)
    def complete(self,messages,max_tokens=2048):
        self._notice_output=min(int(max_tokens),self.budget.max_output_tokens)
        return super().complete(messages,max_tokens)


@dataclass(frozen=True)
class CompletionConfig:
    final_output_ceiling: int = 4096
    recovery_output_tokens: int = 512
    recovery_extra_images: int = 2
    rescue_unknown_with_observed_quote: bool = True


class CompletionEvidenceClient(AssistEvidenceClient):
    def __init__(self,*args,completion_config=None,**kwargs):
        self.completion_config=completion_config or CompletionConfig()
        super().__init__(*args,assist_config=AssistConfig(extra_final_images=0,answer_first=False),**kwargs)
        self.output_allocations=[];self._refusal_recovery=False
    def prepare(self,messages):
        if not self._repairing:return TrajectoryEvidenceClient.prepare(self,messages)
        prepared,images=super().prepare(messages)
        if self._refusal_recovery:
            annotated=copy.deepcopy(prepared)
            annotated[0]['content']+=' If answering instead of UNKNOWN, append Evidence: "an exact short quote from the observed interface text". Otherwise keep UNKNOWN. Never quote the question as evidence.'
            if self.client.can_call(annotated,self.completion_config.recovery_output_tokens):return annotated,images
        return prepared,images
    def complete(self,messages,max_tokens=1024):
        final=not self.is_review(messages) and not self._repairing
        chosen=max_tokens
        if final:
            prepared,_=self.prepare(messages)
            for allowance in sorted({int(max_tokens),2048,self.completion_config.final_output_ceiling},reverse=True):
                if allowance<=self.client.budget.max_output_tokens and self.client.can_call(prepared,allowance):
                    chosen=allowance;break
        response=super().complete(messages,chosen)
        if final:self.output_allocations.append({'physical_request':len(self.client.ledger),'requested_by_controller':max_tokens,'allocated_output':chosen,'baseline_evidence_unchanged':True})
        return response
    def observed_quote(self,reply,images):
        match=re.search(r'Evidence:\s*["“](.{3,240}?)["”]',reply,re.I|re.S)
        if not match:return False
        quote=' '.join(match.group(1).split())
        for im in images:
            if im.get('query_image'):continue
            s=self.store._rows[im['trajectory_id']]['states'][im['state_index']]
            observed=' '.join(str(s.get('accessibility_tree','')).split())
            if quote in observed:return True
        return False


def run_completion_harness(store,question,client):
    result=retrieve_trajectory_evidence(store,question,client)
    original=result['answer'];parsed=boxed_answer(original)
    unknown=parsed is not None and parsed.strip().upper()=='UNKNOWN'
    reason='native_answer_kept';recovery_calls=0;quote_valid=None
    needs_format=bool(original.strip()) and parsed is None
    if needs_format and len(original.strip())<=240 and '\n' not in original.strip():
        result['answer']='\\boxed{'+original.strip()+'}';reason='wrapped_short_native_answer'
    elif client.final_messages and (needs_format or (unknown and client.completion_config.rescue_unknown_with_observed_quote)):
        messages=copy.deepcopy(client.final_messages);previous_config=client.assist_config
        client._repairing=True;client._refusal_recovery=unknown
        client.assist_config=AssistConfig(extra_final_images=client.completion_config.recovery_extra_images,answer_first=True)
        try:
            prepared,images=client.prepare(messages)
            novel_images=any(im.get('supplemental') for im in images)
            if unknown and not novel_images:reason='refusal_kept_no_new_observation'
            elif client.can_call(messages,client.completion_config.recovery_output_tokens):
                reply=client.complete(messages,client.completion_config.recovery_output_tokens);recovery_calls=1
                proposed=boxed_answer(reply)
                actual_images=client.image_events[-1]['images']
                if proposed is None:reason='invalid_recovery_kept_native'
                elif proposed.upper()=='UNKNOWN':reason='refusal_kept_native_candidate'
                elif unknown:
                    quote_valid=client.observed_quote(reply,actual_images)
                    if quote_valid:result['answer']=reply;reason='refusal_rescued_with_observed_quote'
                    else:reason='unanchored_rescue_rejected'
                else:result['answer']=reply;reason='format_recovered'
            else:reason='budget_unavailable_kept_native'
        finally:
            client._repairing=False;client._refusal_recovery=False;client.assist_config=previous_config
    result.update(mode='trajectory_native_completion_harness',completion_config=asdict(client.completion_config),
        native_answer_sha256=text_sha(original),native_non_unknown_box_preserved=(result['answer']==original) if parsed is not None and not unknown else None,
        recovery_status=reason,recovery_calls=recovery_calls,quote_valid=quote_valid,
        output_allocations=client.output_allocations,image_events=client.image_events,assist_events=client.assist_events)
    return result
