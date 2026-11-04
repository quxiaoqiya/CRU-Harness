"""Export whitelisted completion/display receipts after inference, scoring and audit."""
from pathlib import Path
from collections import Counter
import hashlib,json,shutil,sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from long_context_agent.trajectory_memory_assist import boxed_answer
ID='longmemeval_v2_compact_holdout_20261103'
WORK=ROOT/'work'/ID


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def ledger(items):
 keys=['physical_request','prompt_sha256','admission_prompt_tokens','requested_output_tokens','charged_tokens','status','usage_known','finish_reason']
 return [{**{k:r.get(k) for k in keys},'input_tokens':r['usage']['prompt_tokens'] if r.get('usage_known') else None,
          'output_tokens':r['usage']['completion_tokens'] if r.get('usage_known') else None} for r in items]

def summarize(rows):
 n=len(rows);answered=[r for r in rows if r['non_unknown']];correct=sum(r['correct'] is True for r in rows)
 returned=sum(r['status']=='returned' for r in rows)
 return {'n':n,'correct':correct,'accuracy':correct/returned if returned else None,
 'task_success_rate_all_planned':correct/n,'returned':returned,'unknown':sum(r['unknown'] for r in rows),
 'answered':len(answered),'coverage':len(answered)/n,'wrong_answered':sum(r['correct'] is False for r in answered),
 'selective_accuracy':correct/len(answered) if answered else None,
 'tokens':sum(r['tokens'] for r in rows),'calls':sum(r['calls'] for r in rows),
 'deterministic_correct':sum(r['correct'] is True for r in rows if not r['scoring_function'].startswith('llm_')),
 'deterministic_n':sum(not r['scoring_function'].startswith('llm_') for r in rows),
 'judge_correct':sum(r['correct'] is True for r in rows if r['scoring_function'].startswith('llm_')),
 'judge_n':sum(r['scoring_function'].startswith('llm_') for r in rows),
 'image_correct':sum(r['correct'] is True and r['question_image'] for r in rows),
 'image_n':sum(r['question_image'] for r in rows),
 'augmented_cases':sum(r['supplemental_images']>0 for r in rows),
 'repairs':sum(r['recovery_calls'] for r in rows),'judge_failures':sum(r['judge_error'] is not None for r in rows)}


def main():
 o=WORK/'formal';f=json.loads((o/'freeze.json').read_text());scored=json.loads((o/'audit.json').read_text())
 audit=json.loads((WORK/'offline_audit.json').read_text());access=json.loads((o/'scoring_access.json').read_text())
 assert audit['status']=='pass' and not audit['failures'] and audit['replay_passed']==100
 selection=json.loads((WORK/'selection_manifest.json').read_text());meta={r['id']:r for r in selection['selected']}
 assert len(scored['scored'])==audit['scores_recomputed']==100
 rows=[]
 for r in scored['scored']:
  result=r['result']
  rows.append({'id':r['id'],'arm':r['arm'],'domain':r['domain'],'ability':meta[r['id']]['ability'],
   'question_image':r['question_image'],'status':r['status'],'model':r['model'],'correct':r['correct'],
   'unknown':r['unknown'],'non_unknown':r['status']=='returned' and bool(r['parsed_answer'].strip()) and not r['unknown'],
   'scoring_function':r['scoring_function'],'judge_error':r['judge_error'],'error':r['error'],
   'answer_sha256':hashlib.sha256(r['answer'].encode()).hexdigest(),'parsed_answer_sha256':hashlib.sha256(r['parsed_answer'].encode()).hexdigest(),
   'record_sha256':sha(o/'predictions'/f'{r["id"]}__{r["arm"]}.json'),
   'source_sha256':r['source_metadata']['canonical_source_sha256'],'source_text_tokens':r['source_text_tokens_exact'],
   'tokens':r['usage']['charged_tokens'],'calls':r['usage']['physical_requests'],'within_budget':r['usage']['within_budget'],
   'budget':r['usage']['budget'],'ledger':ledger(r['ledger']),'judge_ledger':ledger(r['judge_ledger']),
   'actual_models':sorted({x['actual_model'] for x in r['response_events']}),'completed_at_unix':r['completed_at_unix'],
   'seconds':r['seconds'],'recovery_status':result.get('recovery_status'),
   'final_has_complete_box':boxed_answer(r['answer']) is not None,
   'native_answer_sha256':result.get('native_answer_sha256'),
   'native_box_preserved':result.get('native_non_unknown_box_preserved'),
   'quote_valid':result.get('quote_valid'),
   'output_allocations':result.get('output_allocations',[]),
   'compaction_events':result.get('compaction_events',[]),'recovery_calls':result.get('recovery_calls',0),
   'supplemental_images':sum(e.get('supplemental_images',0) for e in result.get('assist_events',[])),
   'image_occurrences':sum(len(e['images']) for e in result.get('image_events',[])),
   'truncated_generations':sum(e.get('finish_reason')=='length' for e in r['ledger'])})
 old=json.loads((ROOT/'work/longmemeval_v2_qwen_20261103/formal/audit.json').read_text())
 oldmap={r['id']:r for r in old['scored'] if r['arm']=='harness'}
 devstats={};development_freezes={}
 for stage,arm in [('longmemeval_v2_completion_dev_20261103','harness_completion'),('longmemeval_v2_compact_dev_20261103','harness_compact')]:
  devw=ROOT/'work'/stage/'formal';dev=json.loads((devw/'audit.json').read_text());group=[r for r in dev['scored'] if r['arm']==arm]
  development_freezes[arm]=sha(devw/'freeze.json')
  devstats[arm]={'n':len(group),'correct':sum(r['correct'] is True for r in group),
   'old_correct_retained':sum(r['correct'] is True and oldmap[r['id']]['correct'] for r in group),'old_correct_n':8,
   'old_errors_recovered':sum(r['correct'] is True and not oldmap[r['id']]['correct'] for r in group),'old_error_n':12,
   'new_unknown_on_old_correct':sum(r['unknown'] and oldmap[r['id']]['correct'] for r in group)}
 failed=json.loads((ROOT/'work/longmemeval_v2_assist_dev_20261103/formal/audit.json').read_text())
 failed_development={arm:{'n':20,'correct':sum(r['correct'] is True for r in failed['scored'] if r['arm']==arm),
  'old_correct_retained':sum(r['correct'] is True and oldmap[r['id']]['correct'] for r in failed['scored'] if r['arm']==arm)} for arm in ['harness_format','harness_assist']}
 summary={arm:summarize([r for r in rows if r['arm']==arm]) for arm in f['arms']}
 for arm,stats in summary.items():
  group=[r for r in rows if r['arm']==arm];events=[e for r in group for e in r['compaction_events']]
  raw=sum(e['baseline_prompt_tokens'] for e in events);shown=sum(e['display_prompt_tokens'] for e in events)
  stats.update(compacted_cases=sum(bool(r['compaction_events']) for r in group),matched_raw_prompt_tokens=raw,
   matched_display_prompt_tokens=shown,prompt_tokens_saved=raw-shown,matched_prompt_reduction=(raw-shown)/raw if raw else None,
   duplicate_source_chars_removed=sum(w['duplicate_source_chars_removed'] for e in events for w in e['windows']),
   native_box_overrides=sum(r['native_box_preserved'] is False for r in group),
   output_reservations=dict(Counter(str(o['allocated_output']) for r in group for o in r['output_allocations'])))
  enabled=0
  for r in group:
   for event in r['compaction_events']:
    number=event['physical_request'];item=r['ledger'][number-1]
    available=min(r['budget']['context_tokens'],r['budget']['total_tokens']-sum(x['charged_tokens'] for x in r['ledger'][:number-1]))
    enabled+=event['baseline_prompt_tokens']+item['requested_output_tokens']>available>=event['display_prompt_tokens']+item['requested_output_tokens']
  stats['requests_admitted_due_to_display_savings']=enabled
  stats['truncated_generations']=sum(r['truncated_generations'] for r in group)
  stats['unboxed_returned_answers']=sum(r['status']=='returned' and not r['final_has_complete_box'] for r in group)
  stats['recovery_statuses']=dict(Counter(r['recovery_status'] for r in group if r['recovery_status']))

 by={(r['id'],r['arm']):r for r in rows};paired={}
 for arm in ['harness_completion','harness_compact']:
  pairs=[(by[q,'harness']['correct'],by[q,arm]['correct']) for q in f['question_ids']]
  paired[arm]={'both_correct':pairs.count((True,True)),'new_only_correct':pairs.count((False,True)),
   'old_only_correct':pairs.count((True,False)),'neither_correct':pairs.count((False,False)),
   'same_first_review_prompt':sum(by[q,'harness']['ledger'][0]['prompt_sha256']==by[q,arm]['ledger'][0]['prompt_sha256'] for q in f['question_ids'])}
 payload={'schema_version':1,'experiment_id':ID,'benchmark':'LongMemEval-V2','tier':'small','model':f['model'],
 'n_questions':20,'outcomes':100,'n_shared_histories':2,'n_trajectories':200,'n_history_images':5095,'n_query_images':4,
 'arms':f['arms'],'sample_ids':f['question_ids'],'budget':f['budgets'],'judge_budget':f['judge_budget'],'completion_config':f['completion_config'],
 'data_revision':f['data_revision'],'upstream_code_commit':f['upstream_code_commit'],'official_scorer_sha256':f['upstream_scoring_sha256'],
 'wire':{'temperature':0,'seed':f['seed'],'enable_thinking':False,'preserve_thinking':False},
 'question_id_overlap_with_prior40':selection['question_id_overlap'],'question_text_hash_overlap_with_prior40':selection['normalized_question_hash_overlap'],
 'source_histories_reused':True,'selection_seed':selection['seed'],'generation_seed':f['seed'],
 'frozen_at_unix':f['frozen_at_unix'],'first_scoring_access_unix':access['opened_at_unix'],'all100_fixed_before_scoring_label_access':all(r['completed_at_unix']<access['opened_at_unix'] for r in rows),
 'freeze_sha256':sha(o/'freeze.json'),'code_sha256':f['code_sha256'],'summary':summary,'results':rows,'paired':paired,
 'judge_usage':{'calls':sum(len(r['judge_ledger']) for r in rows),
                'tokens':sum(e['charged_tokens'] for r in rows for e in r['judge_ledger'])},
 'source_only_compaction_preflight':json.loads((ROOT/'work/longmemeval_v2_completion_dev_20261103/compaction_preflight.json').read_text()),
 'development':devstats,'development_freeze_sha256':development_freezes,'failed_firstline_development':failed_development,
 'auditor_sha256':audit['auditor_sha256'],'offline_audit_sha256':sha(WORK/'offline_audit.json'),
 'audit':{k:v for k,v in audit.items() if k not in ['receipts','failures']},
 'official_leaderboard_submission':False,'official_LAFS_score':None,
 'judge_substitution':'Qwen instead of official default gpt-5.2; original deterministic functions and judge prompts',
 'native_and_harness_equal_compute':False,'tests_passed':258,'remote_files_modified':False,'github_push':False,
 'limitations':['20 question-disjoint queries, two reused histories; no source-disjoint/general superiority claim',
 'Single run per question/method; provider variation can change answers despite same seed',
 'Same ceilings for Harness variants, larger native Direct ceiling; baseline actual output1024 vs adaptive up to4096; extra original images only in recovery',
 'Mechanical source/budget replay is not independent human entailment validation',
 'No automatic default promotion solely from development results']}
 (ROOT/'benchmarks'/f'{ID}.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
 shutil.copyfile(WORK/'offline_audit.json',ROOT/'benchmarks'/f'{ID}_audit.json')
 report(payload)
 print(json.dumps(summary,indent=2))

def report(p):
 by={(r['id'],r['arm']):r for r in p['results']}
 old=p['summary']['harness'];raw=p['summary']['harness_completion'];new=p['summary']['harness_compact']
 for zh in [True,False]:
  def t(a,b):return a if zh else b
  lines=['# '+t('LongMemEval-V2：原生推理预算与可逆证据展示','LongMemEval-V2: native reasoning budget and reversible evidence display'),' ',
   t(f'新 20 题：直连首尾 {p["summary"]["direct_headtail_native"]["correct"]}/20，原 Harness {old["correct"]}/20，原生完成预算版 {raw["correct"]}/20，可逆展示版 {new["correct"]}/20。',
     f'New 20 questions: head/tail Direct {p["summary"]["direct_headtail_native"]["correct"]}/20, original Harness {old["correct"]}/20, native completion {raw["correct"]}/20, reversible display {new["correct"]}/20.'),'',
   t('问题 ID、规范化正文哈希与此前 40 题零重叠。五种能力各 4 题、两个领域各 10 题，4 题带截图。两份 100 轨迹历史仍复用，共 200 轨迹、5095 历史原图；不是来源独立留出。所有 100 条结果固定后才启用本轮评分标签。',
     'Zero question-ID/normalized-text overlap with the prior 40. Four questions per ability, ten per domain, four query images. Two 100-trajectory histories are reused: 200 trajectories and 5,095 original history images, not a source-disjoint holdout. All 100 outcomes preceded scoring-label access.'),'',
   '| '+t('路径 | 答对／20 | UNKNOWN | 覆盖率 | 已作答准确率 | 输入＋输出 tokens |','Arm | Correct/20 | UNKNOWN | Coverage | Selective accuracy | Input + output tokens |'),'|---|---:|---:|---:|---:|---:|']
  for arm,s in p['summary'].items():
   score='capacity denied' if arm=='direct_full' else f'{s["correct"]}/20 ({s["accuracy"]:.0%})'
   selective='—' if s['selective_accuracy'] is None else f'{s["selective_accuracy"]:.1%}'
   lines.append(f'| {arm} | {score} | {s["unknown"]} | {s["coverage"]:.0%} | {selective} | {s["tokens"]:,} |')
  lines+=['',t('完整直连的容量拒绝不属于语义答错。本轮直连最多 4096 输出，原生请求／累计 262,144 tokens，1 次生成。各 Harness 共享单次 16,384、累计 65,536、8 次生成、单次最多 4096 的硬上限。旧控制器实际请求 1024；完成版在不删证据且可准入时分配 4096／2048／1024。更高输出分配本身就是被测机制，实际用量单列。',
     'Full Direct capacity denials are not semantic wrong answers. Direct allows 4,096 output tokens, 262,144/request and task, one generation. Harness ceilings are shared: 16,384/request, 65,536/task, eight generations, 4,096 maximum output. The original controller requests 1,024; completion allocates 4,096/2,048/1,024 only when admissible without dropping evidence. Output allocation is part of the treatment; actual usage is separate.'),'',
   t('两份完整 canonical JSON 历史文字分别为 31,476,162 和 34,235,595 Qwen tokens，不包含图像像素。Harness 检索的是外部完整来源，模型每次仅接收预算内证据；这些来源长度不表示一次注意力窗口，也不等于语义准确扩展倍数。各路径共同使用按当前请求实际输出额度生成的预算提示。',
     'The two complete canonical JSON history texts contain 31,476,162 and 34,235,595 Qwen tokens, excluding image pixels. Harness retrieves from complete external sources; each model call receives bounded evidence. These source lengths are neither single attention windows nor accurate-context expansion factors. All arms share per-call notices of the actual output reservation.'),'',
   t('模型为 Qwen3.8-Flash-Next-FP8，temperature=0、生成 seed=20261103，enable_thinking=false、preserve_thinking=false。',
     'Model: Qwen3.8-Flash-Next-FP8; temperature=0, generation seed=20261103, enable_thinking=false, preserve_thinking=false.'),'',
   t(f'用量表统计各路径的推理输入＋输出，不包含评分裁判。评分另有独立账本：{p["judge_usage"]["calls"]} 次生成、{p["judge_usage"]["tokens"]:,} tokens，未计入被测方法任务预算。tokenizer 预检与离线重放不作为生成次数。',
     f'Usage tables cover inference input plus output, excluding scoring. Separate judge ledgers record {p["judge_usage"]["calls"]} generations and {p["judge_usage"]["tokens"]:,} tokens, outside method task budgets. Tokenizer preflight and offline replay are not generations.'),'',
   '## '+t('结论与尚未解决的问题','Findings and remaining failures'),' ',
   t(f'两种候选均保留原版答对的 {p["paired"]["harness_completion"]["both_correct"]}/{old["correct"]} 与 {p["paired"]["harness_compact"]["both_correct"]}/{old["correct"]} 题，各新增 {p["paired"]["harness_completion"]["new_only_correct"]} 和 {p["paired"]["harness_compact"]["new_only_correct"]} 道正确题。本轮完成适配组合的准确率比原版提高 {(raw["accuracy"]-old["accuracy"])*100:.0f} 个百分点；这一小样本观察不能唯一归因于某个单独机制。',
     f'The two candidates retain {p["paired"]["harness_completion"]["both_correct"]}/{old["correct"]} and {p["paired"]["harness_compact"]["both_correct"]}/{old["correct"]} baseline-correct questions, adding {p["paired"]["harness_completion"]["new_only_correct"]} and {p["paired"]["harness_compact"]["new_only_correct"]} correct answers. The completion-adaptation bundle improves observed accuracy by {(raw["accuracy"]-old["accuracy"])*100:.0f} percentage points; this pilot cannot isolate one causal mechanism.'),'',
   t(f'完成版与展示版同为 {raw["correct"]}/20，但有 {sum(by[q,"harness_completion"]["correct"]!=by[q,"harness_compact"]["correct"] for q in p["sample_ids"])} 题正确性不同，未做逐题取优拼接。当前证据支持展示层节省空间，不支持宣称展示变换另增准确率。两个新版的视觉子集均只有 {new["image_correct"]}/{new["image_n"]}；本轮没有发生通过新增原图＋短引成功救回 UNKNOWN，拒答语义校准仍未完成。',
     f'Completion and display both score {raw["correct"]}/20, with {sum(by[q,"harness_completion"]["correct"]!=by[q,"harness_compact"]["correct"] for q in p["sample_ids"])} correctness disagreements and no per-question oracle combination. Display saves admission space without an additional observed accuracy gain. Both candidates score only {new["image_correct"]}/{new["image_n"]} on query-image questions. No UNKNOWN answer was rescued through a new image plus quote; semantic abstention calibration remains unresolved.'),'',
   '## '+t('改动与约束','Changes and invariants'),' ',
   t('保留原生推理顺序，不强迫先写答案，不使用自报置信度门控。原检索／review 复用旧实现。资源提示按当前请求真实输出额度计数及生成，避免用较大硬上限误导中间请求。已有非 UNKNOWN boxed 答案不再被二次裁判覆盖。缺失格式时最多补全一次；UNKNOWN 仅在新增状态原图可准入、且新回答附带的短引确实存在于相应观察字段时允许恢复。失败保留原候选。',
     'Preserve native reasoning order, without answer-first forcing or confidence gates. Original retrieval/reviews are reused. Resource notices reflect each actual output request. A valid non-UNKNOWN native box is never overridden by a second judge. One recovery is permitted for missing format; UNKNOWN rescue requires new admitted state images and a short quote present in the corresponding observed field. Failed recovery retains the original candidate.'),'',
   t('可逆展示只用于最终证据：展开 JSON 换行、制表和引号转义；同一来源位置的重叠片段合并为区间并集，保留原窗口别名。不同位置的相同文本不去重，不改变观察的记录身份与数量。展示映射可还原原始来源字符，原图不改；不是生成式摘要或模型注意力窗口扩容。',
     'Final evidence only: render JSON newline/tab/quote escapes and merge coordinate-overlapping ranges with original-header aliases. Identical text at distinct positions is retained, preserving observation identity/multiplicity. Display maps restore original source characters; original images are unchanged. This is not generative summarization or a larger attention window.'),'']
  lines += [t('引用校验只能证明短引存在于观察字段，不能证明该短引足以蕴含答案。UNKNOWN 数量和已作答准确率均公开；当前实验没有独立人工的视觉相关性标注，不能把图片完整性核对称为视觉语义召回率提升。',
    'Quote checks establish presence in an observed field, not answer entailment. UNKNOWN counts and selective accuracy are reported. There are no independent human visual-relevance labels, so image integrity is not a visual semantic recall improvement.'),'']
  if new['matched_raw_prompt_tokens']:
   lines += [t(f'实际发生展示变换的完整提示，匹配计数合计 {new["matched_raw_prompt_tokens"]:,} → {new["matched_display_prompt_tokens"]:,} tokens，节省 {new["prompt_tokens_saved"]:,}（{new["matched_prompt_reduction"]:.2%}）。其中 {new["requests_admitted_due_to_display_savings"]} 次请求的原表示在相同输出保留量下无法准入，展示后可以准入。删除的重复来源字符为 {new["duplicate_source_chars_removed"]:,}；若为零，不能将收益归因于重叠合并。',
     f'Matched complete prompts with display transforms total {new["matched_raw_prompt_tokens"]:,} → {new["matched_display_prompt_tokens"]:,} tokens: {new["prompt_tokens_saved"]:,} saved ({new["matched_prompt_reduction"]:.2%}). On {new["requests_admitted_due_to_display_savings"]} requests the raw representation would not fit with the same output reservation. Duplicate coordinate characters removed: {new["duplicate_source_chars_removed"]:,}; zero means no measured overlap-dedup benefit.'),'']
   lines += [t('该比例仅针对发生展示变换的匹配完整提示，不是端到端总用量下降率。追检次数、证据路径和实际输出可能变化，端到端输入＋输出用量以上表为准。',
     'This ratio covers matched complete prompts that were transformed, not end-to-end total-token reduction. Retrieval rounds, evidence paths and actual outputs can differ; the table above reports end-to-end input plus output usage.'),'']
   lines += [t('完整提示使用准入计数，各含 16 tokens 安全余量；配对差值不受共同余量影响。来源文字预检是单独的 tokenizer 原始计数。',
     'Complete-prompt admission counts each include a 16-token safety margin, which cancels in paired differences. Source-text preflight uses separate raw tokenizer counts.'),'']
  lines += [t('默认轨迹控制器保持原样；新增完成／展示控制器为显式实验入口，不根据这 20 题自动选路。每种方法每题只运行一次，没有逐题取优、投票挑选或测试后改写答案。',
    'The default trajectory controller remains unchanged. Completion/display controllers are explicit experimental entry points, without routing learned from these 20 questions. One run per question/method; no per-question best-of selection, voting, or post-test answer edits.'),'',
   '| Candidate vs original | Both correct | Candidate only | Original only | Neither |','|---|---:|---:|---:|---:|']
  for arm,s in p['paired'].items():lines.append(f'| {arm} | {s["both_correct"]} | {s["new_only_correct"]} | {s["old_only_correct"]} | {s["neither_correct"]} |')
  lines+=['','| Candidate | Native valid boxes overridden | Recovery calls | Supplemental image cases | Final output reservations |','|---|---:|---:|---:|---|']
  for arm in ['harness_completion','harness_compact']:
   s=p['summary'][arm];lines.append(f'| {arm} | {s["native_box_overrides"]} | {s["repairs"]} | {s["augmented_cases"]} | {json.dumps(s["output_reservations"],sort_keys=True)} |')
  lines+=['']
  lines += ['| Arm | Length-truncated generations | Unboxed final answers | Recovery outcomes |','|---|---:|---:|---|']
  for arm in ['harness','harness_completion','harness_compact']:
   s=p['summary'][arm];lines.append(f'| {arm} | {s["truncated_generations"]} | {s["unboxed_returned_answers"]} | {json.dumps(s["recovery_statuses"],sort_keys=True)} |')
  lines+=['']
  lines+=['## '+t('开发记录与正确题保留','Development and positive retention'),'','| Candidate | Correct/20 | Old correct retained/8 | Old errors recovered/12 |','|---|---:|---:|---:|']
  for arm,s in p['development'].items():lines.append(f'| {arm} | {s["correct"]} | {s["old_correct_retained"]} | {s["old_errors_recovered"]} |')
  for arm,s in p['failed_firstline_development'].items():lines.append(f'| failed {arm} | {s["correct"]} | {s["old_correct_retained"]} | — |')
  lines+=['',t('开发集复用首轮 20 题，原版 8 对、12 错；这些数字只用于开发，不能当作留出成绩。强迫先答方案的失败记录保留。另一个开发样本仅对来源文字计数，10383→9357 tokens、节省 9.88%，不是完整多模态提示的压缩比。',
     'Development reuses the original 20 (eight correct, twelve wrong), not holdout scores. Failed answer-first trials are retained. A separate source-text-only preflight measures 10,383→9,357 tokens (9.88%), not a complete multimodal-prompt ratio.'),'',
   '## '+t('分项和逐题结果','Subsets and per-question outcomes'),'','| Subset | Direct | Original | Completion | Compact |','|---|---:|---:|---:|---:|']
  for label,k,n in [('Deterministic','deterministic_correct','deterministic_n'),('Qwen judged','judge_correct','judge_n'),('Query image','image_correct','image_n')]:
   lines.append('| '+label+' | '+' | '.join(f'{p["summary"][a][k]}/{p["summary"][a][n]}' for a in ['direct_headtail_native','harness','harness_completion','harness_compact'])+' |')
  for field in ['ability','domain']:
   for label in sorted({r[field] for r in p['results']}):
    cells=[]
    for arm in ['direct_headtail_native','harness','harness_completion','harness_compact']:
     group=[r for r in p['results'] if r['arm']==arm and r[field]==label]
     cells.append(f'{sum(r["correct"] for r in group)}/{len(group)}')
    lines.append('| '+field+': '+label+' | '+' | '.join(cells)+' |')
  lines+=['','| ID | Domain | Ability | Image | Direct | Original | Completion | Compact |','|---|---|---|---|---|---|---|---|']
  for q in p['sample_ids']:
   r=by[q,'harness'];cells=[]
   for arm in ['direct_headtail_native','harness','harness_completion','harness_compact']:
    x=by[q,arm];cells.append('correct' if x['correct'] else 'UNKNOWN' if x['unknown'] else 'incorrect')
   lines.append(f'| {q} | {r["domain"]} | {r["ability"]} | {r["question_image"]} | '+' | '.join(cells)+' |')
  lines+=['','## '+t('审计和适用范围','Audit and scope'),' ',
   t(f'258 项检查通过。离线审计 {p["audit"]["ledgers_passed"]}/100 账本、{p["audit"]["scores_recomputed"]}/100 评分重算、{p["audit"]["replay_passed"]}/100 控制流程重放，状态 {p["audit"]["status"]}。核对原图、原位置区间并集、可逆还原及真实准入／输出额度，不等于独立人工语义验证。',
     f'258 checks passed. Offline audit: {p["audit"]["ledgers_passed"]}/100 ledgers, {p["audit"]["scores_recomputed"]}/100 score re-parses, {p["audit"]["replay_passed"]}/100 controller replays; status {p["audit"]["status"]}. Original images, source-position union, round trips and actual output/admission are checked, not independent human entailment.'),'',
   t('单次 20 题、两份共享历史不证明普遍显著优势。固定 seed 和温度 0 仍可能产生服务端输出变化。各 Harness 硬上限相同、实际输入与输出分配不同；直连预算更大。原模型、原控制器和此前结果未覆盖，未改原环境依赖或远端文件。官方确定性评分和裁判提示固定，LLM 裁判使用 Qwen 代替默认 gpt-5.2；不是官方 LAFS 或线上榜单分数。',
     'One run on 20 questions/two shared histories does not establish general significant superiority. Fixed seed/temperature zero may still vary. Harness hard ceilings match; actual input/output allocations differ, and Direct has a larger budget. Original controllers/results, dependencies and remote files remain unchanged. Pinned official deterministic scoring/prompts are used, with Qwen replacing default gpt-5.2 as judge; not official LAFS or an online leaderboard score.'),'',
   '[Official code](https://github.com/xiaowu0162/LongMemEval-V2) · [Paper](https://arxiv.org/abs/2605.12493) · [Dataset](https://huggingface.co/datasets/xiaowu0162/longmemeval-v2)','',
   f'Data `{p["data_revision"]}`; code `{p["upstream_code_commit"]}`; freeze `{p["freeze_sha256"]}`.','',
   '```bash','python3 -S scripts/verify_longmemeval_v2_compact_results.py','```','',
   f'[Results JSON](../benchmarks/{ID}.json) · [Offline audit](../benchmarks/{ID}_audit.json) · [Explicit opt-in API](trajectory_completion_experiments.md)','']
  (ROOT/'docs'/f'{ID}{".zh-CN" if zh else ""}.md').write_text('\n'.join(lines))


if __name__=='__main__':main()
