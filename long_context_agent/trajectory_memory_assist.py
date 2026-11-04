"""Answer-preserving trajectory assistance, opt-in and baseline-compatible.

The legacy retrieval/review controller is reused without edits. Optional final
images must fit without removing a baseline text block or baseline image. There
is no confidence gate and a failed repair never erases the original candidate.
"""
from dataclasses import asdict, dataclass
import copy,json,re
from .adaptive_evidence import EvidenceWindow,parse_review,validate_supported_spans
from .trajectory_memory import (TrajectoryEvidenceClient,_HEADER,text_sha,
                                retrieve_trajectory_evidence)


@dataclass(frozen=True)
class AssistConfig:
    extra_final_images: int = 2
    answer_first: bool = True
    repair_tokens: int = 384
    max_repairs: int = 1


def boxed_answer(text):
    """Read the last balanced box; never infer an answer from a partial box."""
    answers=[]
    for match in re.finditer(r'\\boxed\s*\{',str(text)):
        depth=1;start=match.end()
        for i in range(start,len(text)):
            if text[i]=='{':depth+=1
            elif text[i]=='}':
                depth-=1
                if depth==0:answers.append(text[start:i]);break
    return answers[-1].strip() if answers else None


class AssistEvidenceClient(TrajectoryEvidenceClient):
    def __init__(self,*args,assist_config=None,**kwargs):
        super().__init__(*args,**kwargs)
        self.assist_config=assist_config or AssistConfig()
        self.supported_pairs=[];self.review_images=[];self.assist_events=[];self._prepared_cache={}
        self.final_messages=None;self._repairing=False

    @staticmethod
    def is_review(messages):
        return 'evidence-gap reviewer' in str(messages[0].get('content',''))

    def extra_candidates(self,base_images):
        if self.assist_config.extra_final_images<=0:return []
        base_pairs={(r.get('trajectory_id'),r.get('state_index')) for r in base_images}
        seen_hash={r['image_sha256'] for r in base_images}
        candidates={}
        # Retain images for supported source regions, including earlier reviews.
        for pair in self.supported_pairs:
            for state in self.store.images_for_spans([pair],limit=8):
                key=state.trajectory_id,state.state_index
                if key not in base_pairs:candidates[key]=(100,state,'prior_support')
        # A neighboring original state can disambiguate an action's effect.
        temporal=bool(re.search(r'\b(after|before|click|submit|when|workflow|steps|actions)\b',self.question_text or '',re.I))
        if temporal:
            bindings={(x.trajectory_id,x.state_index):x for x in self.store.bindings}
            for tid,i in base_pairs:
                if tid is None:continue
                for delta in (-1,1):
                    key=tid,i+delta
                    if key in bindings and key not in base_pairs and key not in candidates:
                        candidates[key]=(40,bindings[key],'adjacent_state')
        out=[]
        for _,state,reason in sorted(candidates.values(),key=lambda x:(-x[0],x[1].start)):
            if not state.image:continue
            payload,sha=self.store.image_payload(state.image)
            if sha in seen_hash:continue
            seen_hash.add(sha);out.append((state,payload,sha,reason))
            if len(out)>=self.assist_config.extra_final_images:break
        return out if self.assist_config.extra_final_images>0 else []

    def prepare(self,messages):
        if self.is_review(messages):return super().prepare(messages)
        key=(text_sha(json.dumps(messages,sort_keys=True,ensure_ascii=False)),self.remaining_tokens,self.remaining_calls,self._repairing)
        if key in self._prepared_cache:return self._prepared_cache[key]
        original,original_images=super().prepare(messages)
        ceiling=min(self.context_tokens,self.remaining_tokens)
        output=self.store.config.answer_output_tokens
        # Do not spend augmentation tokens before the original request fits.
        baseline_count=self.client.count_messages(original)
        if baseline_count+output>ceiling:
            self._prepared_cache[key]=(original,original_images);return original,original_images
        extras=self.extra_candidates(original_images)
        policy=original[0]['content']
        if self.assist_config.answer_first or self._repairing:
            policy+='\nPut exactly one final \\boxed{answer} on the FIRST line. Give at most one short explanatory sentence afterward if needed. If the question has a false premise, state the correction in the box. If evidence is insufficient, use \\boxed{UNKNOWN}. Do not replace observed labels with familiar synonyms.'
        if self._repairing:policy+=' FORMAT RECOVERY: return exactly the boxed answer and nothing else.'
        while True:
            prepared=copy.deepcopy(original);prepared[0]['content']=policy
            parts=prepared[-1]['content'];insert=next((i for i,p in enumerate(parts) if p.get('type')=='text' and p['text'].startswith('\n\n### Question to answer:')),len(parts))
            extra_parts=[];receipts=[r for r in original_images if not r.get('query_image')]
            for state,payload,sha,reason in extras:
                observed=self.store._rows[state.trajectory_id]['states'][state.state_index]
                label=f'Additional original screenshot: trajectory {state.trajectory_id}, state {state.state_index}; source state chars {state.start}:{state.end}. '
                label+='Observed URL and action leading to this state: '+json.dumps({'url':observed.get('url'),'action':observed.get('action')},ensure_ascii=False)
                extra_parts += [{'type':'text','text':label},{'type':'image_url','image_url':{'url':payload}}]
                receipts.append({'trajectory_id':state.trajectory_id,'state_index':state.state_index,
                    'source_state_span':[state.start,state.end],'image_sha256':sha,'supplemental':True,'selection_reason':reason})
            parts[insert:insert]=extra_parts
            receipts.extend(r for r in original_images if r.get('query_image'))
            admitted=self.client.count_messages(prepared)
            if admitted+output<=ceiling:break
            if extras:extras.pop();continue
            # The answer-first instruction is optional too; preserve a fitting baseline.
            prepared,receipts=original,original_images;admitted=baseline_count;break
        self._prepared_cache[key]=(prepared,receipts)
        return prepared,receipts

    def complete(self,messages,max_tokens=1024):
        review=self.is_review(messages)
        if not review:self.final_messages=copy.deepcopy(messages)
        response=super().complete(messages,max_tokens)
        if review:
            spans=list(dict.fromkeys((int(a),int(b)) for a,b in _HEADER.findall(str(messages[-1]['content']))))
            windows=[EvidenceWindow(a,b,self.store.context[a:b],'observed',0) for a,b in spans]
            good,_=validate_supported_spans(parse_review(response).get('supported_spans'),windows)
            for text in good:
                a,b=map(int,re.findall(r'\d+',text))
                if (a,b) not in self.supported_pairs:self.supported_pairs.append((a,b))
            self.review_images.extend(self.image_events[-1]['images'] if self.image_events else [])
        else:
            # The event below reports the actual physical request, not a new candidate preparation.
            actual=self.image_events[-1] if self.image_events else {}
            self.assist_events.append({'physical_request':len(self.client.ledger),
                'supplemental_images':sum(bool(r.get('supplemental')) for r in actual.get('images',[])),
                'baseline_images_retained':True,'baseline_source_text_retained':True})
        return response


def run_assisted_trajectory(store,question,client):
    result=retrieve_trajectory_evidence(store,question,client)
    original=result['answer'];recovery='not_needed';repairs=0
    if original.strip() and boxed_answer(original) is None:
        # Preserve an already-short native answer; no semantic rewriting.
        if len(original.strip())<=240 and '\n' not in original.strip():
            result['answer']='\\boxed{'+original.strip()+'}';recovery='wrapped_short_native_answer'
        elif client.final_messages and client.assist_config.max_repairs and client.remaining_calls:
            repair=copy.deepcopy(client.final_messages)
            repair[0]['content']='Answer from the supplied original evidence. Return exactly one boxed answer on the first line and no other text. Correct a false premise explicitly; use UNKNOWN only if the evidence does not establish an answer.'
            client._repairing=True
            try:
                if client.can_call(repair,client.assist_config.repair_tokens):
                    candidate=client.complete(repair,client.assist_config.repair_tokens);repairs=1
                    if boxed_answer(candidate) is not None:result['answer']=candidate;recovery='fresh_boxed_repair'
                    else:recovery='invalid_repair_kept_original'
                else:recovery='no_repair_budget_kept_original'
            finally:client._repairing=False
    result.update(mode='trajectory_answer_preserving_assist',assist_config=asdict(client.assist_config),
        format_recovery=recovery,repair_calls=repairs,assist_events=client.assist_events,
        supported_image_source_spans=[list(x) for x in client.supported_pairs],image_events=client.image_events)
    return result
