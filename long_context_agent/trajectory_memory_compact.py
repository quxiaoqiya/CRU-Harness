"""Reversible display of JSON string escapes in final evidence only.

Raw source spans are retained and address the original canonical source. This
is presentation, not summarization: every edit can be reversed byte for byte.
"""
from dataclasses import asdict,dataclass
import copy,json,re
from .trajectory_memory import _HEADER,text_sha,TrajectoryEvidenceClient
from .trajectory_memory_completion import CompletionEvidenceClient,run_completion_harness


_ESCAPE=re.compile(r'\\(?:\\|n|t|")')


def render_source_range(source,start,end):
    raw=source[start:end];out=[];edits=[];cursor=0;length=0
    # An escape begun before this window is never partially decoded.
    preceding=0;i=start-1
    while i>=0 and source[i]=='\\':preceding+=1;i-=1
    scan_from=1 if preceding%2 else 0
    for match in _ESCAPE.finditer(raw,scan_from):
        token=match.group()
        if token=='\\\\':continue
        decoded={'\\n':'\n','\\t':'\t','\\"':'"'}[token]
        literal=raw[cursor:match.start()];out.append(literal);length+=len(literal)
        edits.append({'display_offset':length,'display':decoded,'original':token})
        out.append(decoded);length+=len(decoded);cursor=match.end()
    out.append(raw[cursor:]);view=''.join(out)
    assert restore_source_view(view,edits)==raw,'Non-reversible display transform'
    return view,edits


def restore_source_view(view,edits):
    restored=view
    for edit in reversed(edits):
        p=edit['display_offset'];shown=edit['display']
        if restored[p:p+len(shown)]!=shown:raise ValueError('Display edit does not match')
        restored=restored[:p]+edit['original']+restored[p+len(shown):]
    return restored


def compact_evidence_part(text,source):
    blocks=[]
    for index,header in enumerate(_HEADER.finditer(text)):
        a,b=map(int,header.groups());body_start=text.find('\n',header.end())+1
        if body_start<=0:raise ValueError('Missing evidence header newline')
        body_end=body_start+(b-a)
        if text[body_start:body_end]!=source[a:b]:raise ValueError('Evidence body is not the original source range')
        blocks.append({'a':a,'b':b,'index':index,'header_start':header.start(),
                       'body_start':body_start,'body_end':body_end,'header':text[header.start():body_start]})
    if not blocks:return text,[]
    can_merge=all(not text[x['body_end']:y['header_start']].strip() for x,y in zip(blocks,blocks[1:]))
    groups=[]
    if can_merge:
        for block in sorted(blocks,key=lambda x:(x['a'],x['b'])):
            if groups and block['a']<=groups[-1]['b']:
                groups[-1]['b']=max(groups[-1]['b'],block['b']);groups[-1]['aliases'].append(block)
            else:groups.append({'a':block['a'],'b':block['b'],'aliases':[block]})
        groups.sort(key=lambda g:min(x['index'] for x in g['aliases']))
    else:groups=[{'a':x['a'],'b':x['b'],'aliases':[x]} for x in blocks]
    rendered=[];receipts=[];replacements=[]
    for g in groups:
        a,b=g['a'],g['b'];raw=source[a:b];view,edits=render_source_range(source,a,b)
        aliases=sorted(g['aliases'],key=lambda x:x['index'])
        if len(aliases)==1:head=aliases[0]['header']
        else:
            reference=json.dumps([{'span':[x['a'],x['b']],'header':x['header'].strip()} for x in aliases],ensure_ascii=False)
            head=f'[source_chars {a}:{b} | gap overlap_union | aliases {reference}]\n'
        rendered.append(head+view)
        receipts.append({'source_span':[a,b],'original_source_spans':[[x['a'],x['b']] for x in aliases],
                         'raw_sha256':text_sha(raw),'display_sha256':text_sha(view),'edit_count':len(edits),
                         'duplicate_source_chars_removed':sum(x['b']-x['a'] for x in aliases)-(b-a),
                         'reversible':restore_source_view(view,edits)==raw})
        if not can_merge:replacements.append((aliases[0]['body_start'],aliases[0]['body_end'],view))
    if can_merge:
        result=text[:blocks[0]['header_start']]+'\n\n'.join(rendered)+text[blocks[-1]['body_end']:]
    else:
        result=text
        for a,b,view in reversed(replacements):result=result[:a]+view+result[b:]
    return result,receipts


class CompactCompletionClient(CompletionEvidenceClient):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.compaction_events=[];self._display_receipts={};self._count_cache={}
    def prepare(self,messages):
        baseline,base_images=TrajectoryEvidenceClient.prepare(self,messages)
        if self.is_review(messages):return baseline,base_images
        extras=self.extra_candidates(base_images) if self._repairing else []
        while True:
            candidate=copy.deepcopy(baseline);images=[im for im in base_images if not im.get('query_image')]
            if self._repairing:
                candidate[0]['content']+=' FORMAT RECOVERY: return exactly one boxed answer on the first line. Correct false premises explicitly and keep UNKNOWN when evidence is insufficient.'
                if self._refusal_recovery:candidate[0]['content']+=' A non-UNKNOWN answer must append Evidence: "one exact short quote from observed interface text"; never quote the question.'
            parts=candidate[-1]['content'];insert=next((i for i,part in enumerate(parts) if part.get('type')=='text' and part['text'].startswith('\n\n### Question to answer:')),len(parts))
            added=[]
            for state,payload,sha,reason in extras:
                observed=self.store._rows[state.trajectory_id]['states'][state.state_index]
                label=f'Additional original screenshot: trajectory {state.trajectory_id}, state {state.state_index}; source state chars {state.start}:{state.end}. '
                label+='Observed URL and action: '+json.dumps({'url':observed.get('url'),'action':observed.get('action')},ensure_ascii=False)
                added += [{'type':'text','text':label},{'type':'image_url','image_url':{'url':payload}}]
                images.append({'trajectory_id':state.trajectory_id,'state_index':state.state_index,'source_state_span':[state.start,state.end],
                               'image_sha256':sha,'supplemental':True,'selection_reason':reason})
            parts[insert:insert]=added;images.extend(im for im in base_images if im.get('query_image'))
            prepared=copy.deepcopy(candidate);receipts=[]
            for part in prepared[-1]['content']:
                if part.get('type')=='text' and _HEADER.search(part['text']):
                    try:part['text'],rs=compact_evidence_part(part['text'],self.store.context)
                    except ValueError:return candidate,images
                    receipts.extend(rs)
            if receipts:prepared[0]['content']+=' JSON newline, tab and quoted-text escapes are displayed literally. Overlapping ranges share one copy with original-header aliases. No source observation or distinct source position is discarded.'
            key=(text_sha(json.dumps(prepared,sort_keys=True,ensure_ascii=False)),self.remaining_tokens,self.remaining_calls,getattr(self.client,'_notice_output',None))
            if key not in self._count_cache:self._count_cache[key]=(self.client.count_messages(candidate),self.client.count_messages(prepared))
            raw_tokens,view_tokens=self._count_cache[key]
            if view_tokens>=raw_tokens:prepared=candidate;view_tokens=raw_tokens;receipts=[]
            if not self._repairing or view_tokens+self.completion_config.recovery_output_tokens<=min(self.context_tokens,self.remaining_tokens) or not extras:break
            extras.pop()
        if receipts:
            fingerprint=text_sha(json.dumps(prepared,sort_keys=True,ensure_ascii=False))
            self._display_receipts[fingerprint]={'windows':receipts,'baseline_prompt_tokens':raw_tokens,
                'display_prompt_tokens':view_tokens,'saved_prompt_tokens':raw_tokens-view_tokens}
        return prepared,images
    def complete(self,messages,max_tokens=1024):
        before=len(self.client.ledger)
        reply=super().complete(messages,max_tokens)
        if not self.is_review(messages) and len(self.client.ledger)>before:
            event=self.image_events[-1];receipt=self._display_receipts.get(event['prepared_messages_sha256'])
            if receipt:self.compaction_events.append({'physical_request':len(self.client.ledger),**receipt})
        return reply


def run_compact_completion(store,question,client):
    result=run_completion_harness(store,question,client)
    result.update(mode='trajectory_reversible_compact_completion',compaction_events=client.compaction_events,
                  compaction_is_lossless_display_not_summary=True)
    return result
