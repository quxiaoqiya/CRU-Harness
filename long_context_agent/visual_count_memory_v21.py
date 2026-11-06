"""ID-explicit visual facts and lossless, label-blind response normalization.

The frozen v2 experiment remains unchanged. This module also supports replay
of its successful paid responses without synthesizing missing observations.
"""
from dataclasses import dataclass
import math
import re

from .multimodal_page_harness_v2 import CountFacts, count_prompt, json_values


ALIASES = {'figures': ('figures','figure','pictures','picture','figure_count','picture_count'),
           'tables': ('tables','table','table_count'), 'count': ('count','number','total')}


def integer(value):
    if type(value) is int and value >= 0:return value
    if type(value) is float and math.isfinite(value) and value >= 0 and value.is_integer():return int(value)
    if isinstance(value,str) and re.fullmatch(r'\d+',value.strip()):return int(value)
    return None


def values(record,fields):
    if isinstance(record,dict):
        if isinstance(record.get('counts'),(list,tuple)):
            record=record['counts']
        else:
            lower={str(k).lower():v for k,v in record.items()}
            output={}
            for field in fields:
                choices=[lower[k] for k in ALIASES.get(field,(field,)) if k in lower]
                if not choices:return None
                normalized=[integer(v) for v in choices]
                if any(v is None for v in normalized) or len(set(normalized))!=1:return None
                output[field]=normalized[0]
            return output
    vector=record if isinstance(record,(list,tuple)) else [record]
    if len(vector)!=len(fields):return None
    normalized=[integer(v) for v in vector]
    if any(v is None for v in normalized):return None
    return dict(zip(fields,normalized))


def normalize_counts(raw,ids,fields=('figures','tables')):
    """Anonymous rows require exact length; explicit IDs allow partial output."""
    ids=list(ids);allowed=set(ids);best=[]
    def rows(value):
        if isinstance(value,dict):
            for key in ['pages','results','counts']:
                if key in value and isinstance(value[key],(list,dict)):return rows(value[key])
            if all(field in value and isinstance(value[field],list) and len(value[field])==len(ids) for field in fields):
                return [{'id':key,**dict(zip(fields,vector))} for key,vector in zip(ids,zip(*(value[f] for f in fields)))]
            if any(key in allowed for key in value):
                return [{'id':key,**v} for key,record in value.items() if key in allowed and (v:=values(record,fields)) is not None]
            return []
        if not isinstance(value,list):return []
        explicit=[];has_explicit=False
        for item in value:
            if isinstance(item,dict):
                key=item.get('id',item.get('page_id',item.get('source_id')))
                if key is not None:
                    has_explicit=True
                    if key in allowed and (v:=values(item,fields)) is not None:explicit.append({'id':key,**v})
            elif isinstance(item,list) and len(item)==len(fields)+1 and isinstance(item[0],str):
                has_explicit=True
                if item[0] in allowed and (v:=values(item[1:],fields)) is not None:explicit.append({'id':item[0],**v})
        if has_explicit:return explicit
        if len(value)!=len(ids):return []
        return [{'id':key,**v} for key,item in zip(ids,value) if (v:=values(item,fields)) is not None]
    for candidate in json_values(raw):
        normalized=[]
        for row in rows(candidate):
            v=values(row,fields)
            if row.get('id') in allowed and v is not None:normalized.append({'id':row['id'],**v})
        if len({r['id'] for r in normalized})>len({r['id'] for r in best}):best=normalized
        if len({r['id'] for r in best})==len(ids):break
    return best


@dataclass(frozen=True)
class CountMemoryConfig:
    batch_pages:int=16
    repair_pages:int=8
    max_attempts:int=3
    max_recovery_calls:int=64
    output_tokens:int=2048


def explicit_prompt(question,fields):
    definition=count_prompt(question,fields).split(' Return JSON ',1)[0]
    return (definition+' Return one compact JSON object {"pages":[{"id":"EXACT supplied page ID",'
            +','.join('"'+f+'":0' for f in fields)+'}]}. Include every supplied ID exactly once. '
            'Never omit or reorder identity. Use nonnegative integers, or null for an unreadable count. '
            'Do not return an anonymous position-only list. Question: '+question)


def run_count_memory(question,index,page_ids,client,*,fields=('figures','tables'),config=None,on_checkpoint=None):
    if re.search(r'\b(?:distinct|unique)\b',question,re.I):
        raise ValueError('Non-additive cross-page identity counting requires an entity operator, not a sum')
    config=config or CountMemoryConfig();ids=list(dict.fromkeys(page_ids))
    facts=CountFacts(ids,fields);pixels=set();attempts={key:0 for key in ids};trace=[]
    def inspect(batch,reviewed=False):
        parts=[]
        for key in batch:
            parts += [{'type':'text','text':'Source page ID: '+key},
                      {'type':'image_url','image_url':{'url':index.image(index.pages[key])}}]
        messages=[{'role':'system','content':explicit_prompt(question,fields)},{'role':'user','content':parts}]
        if not client.can_call(messages,config.output_tokens):return False
        raw=client.complete(messages,config.output_tokens)
        pixels.update(batch)
        for key in batch:attempts[key]+=1
        accepted=[]
        for row in normalize_counts(raw,batch,fields):
            if facts.observe(row['id'],row,exposed=pixels,physical_request=len(client.ledger),reviewed=reviewed):accepted.append(row['id'])
        trace.append({'physical_request':len(client.ledger),'page_ids':batch,'reviewed':reviewed,
                      'accepted_fact_ids':sorted(set(accepted)),'empty_content':not bool(raw.strip())})
        if on_checkpoint:on_checkpoint({'aggregation':facts.summary(),'trace':trace,'pixel_page_ids':sorted(pixels)})
        cache=getattr(index,'_images',None)
        if isinstance(cache,dict):
            for key in batch:cache.pop(key,None)
        return True
    for start in range(0,len(ids),config.batch_pages):
        if not inspect(ids[start:start+config.batch_pages]):break
    recoveries=0;repair_size=config.repair_pages
    while client.remaining_calls and recoveries<config.max_recovery_calls:
        summary=facts.summary()
        missing=[key for key in ids if (key in summary['missing_ids'] or key in summary['conflict_ids']) and attempts[key]<config.max_attempts]
        if not missing:break
        missing.sort(key=lambda key:(attempts[key],ids.index(key)))
        before=len(facts.facts)
        if not inspect(missing[:repair_size],reviewed=True):break
        recoveries+=1
        if len(facts.facts)==before:repair_size=max(1,repair_size//2)
        else:repair_size=config.repair_pages
    summary=facts.summary();answer=str(sum(summary['totals'].values())) if summary['complete'] else 'UNKNOWN'
    return {'answer':answer,'structured_answer':summary['totals'],'aggregation':summary,'trace':trace,
        'pixel_page_ids':sorted(pixels),'read_page_ids':sorted(pixels),'final_fact_page_ids':sorted(facts.facts),
        'recovery_calls':recoveries,'attempts':attempts,
        'stop_reason':'deterministic_complete_aggregation' if summary['complete'] else 'bounded_missing_facts_UNKNOWN',
        'route':'id_explicit_visual_count_memory_v21'}
