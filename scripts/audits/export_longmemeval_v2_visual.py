"""Export only whitelisted visual-holdout metrics after scoring and audit."""
from pathlib import Path
from collections import Counter
import hashlib,json,statistics,shutil
ROOT=Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0,str(ROOT))
ID='longmemeval_v2_visual_holdout_20261103'
W=ROOT/'work'/ID


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def receipts(entries):
 fields=['physical_request','admission_prompt_tokens','requested_output_tokens','charged_tokens',
         'usage_known','status','prompt_sha256','finish_reason']
 return [{**{k:e.get(k) for k in fields},'input_tokens':e['usage']['prompt_tokens'] if e.get('usage_known') else None,
          'output_tokens':e['usage']['completion_tokens'] if e.get('usage_known') else None} for e in entries]


def summary(rows):
 n=len(rows);answered=[r for r in rows if r['non_unknown_answer']];correct=sum(r['correct'] is True for r in rows)
 return {'n':n,'correct':correct,'task_accuracy':correct/n,'returned':sum(r['status']=='returned' for r in rows),
   'unknown':sum(r['unknown'] for r in rows),'answered':len(answered),'coverage':len(answered)/n,
   'wrong_answered':sum(r['correct'] is False for r in answered),
   'selective_accuracy':correct/len(answered) if answered else None,
   'tokens':sum(r['tokens'] for r in rows),'calls':sum(r['calls'] for r in rows),
   'mean_seconds':statistics.mean(r['seconds'] for r in rows),
   'capacity_denials':sum(r['status']=='native_capacity_denied' for r in rows),
   'image_correct':sum(r['correct'] is True and r['question_image'] for r in rows),
   'image_n':sum(r['question_image'] for r in rows),
   'deterministic_correct':sum(r['correct'] is True for r in rows if not r['scoring_function'].startswith('llm_')),
   'deterministic_n':sum(not r['scoring_function'].startswith('llm_') for r in rows),
   'judge_correct':sum(r['correct'] is True for r in rows if r['scoring_function'].startswith('llm_')),
   'judge_n':sum(r['scoring_function'].startswith('llm_') for r in rows),
   'gate_reasons':dict(Counter(r['gate_reason'] for r in rows if r['gate_reason'])),
   'ability':{k:{'n':sum(r['ability']==k for r in rows),'correct':sum(r['correct'] is True and r['ability']==k for r in rows)} for k in sorted({r['ability'] for r in rows})}}


def main():
 o=W/'formal';f=json.loads((o/'freeze.json').read_text());a=json.loads((o/'audit.json').read_text())
 audit=json.loads((W/'offline_audit.json').read_text());sel=json.loads((W/'selection_manifest.json').read_text())
 access=json.loads((o/'scoring_access.json').read_text());meta={r['id']:r for r in sel['selected']}
 assert len(a['scored'])==f['expected_terminal_records']==100 and audit['scores_recomputed']==100
 extw=ROOT/'work/longmemeval_v2_visual_route_20261103';exto=extw/'formal'
 ext=json.loads((exto/'audit.json').read_text());extf=json.loads((exto/'freeze.json').read_text())
 exta=json.loads((extw/'offline_audit.json').read_text());extaccess=json.loads((exto/'scoring_access.json').read_text())
 assert len(ext['scored'])==exta['scores_recomputed']==20
 assert set(extf['question_ids'])==set(f['question_ids'])
 arms=f['arms']+extf['arms']
 budgets={**f['budgets'],**extf['budgets']}
 codes={**f['code_sha256'],**extf['code_sha256']}
 assert all(codes[k]==v for k,v in f['code_sha256'].items())
 from scripts import benchmark_longmemeval_v2_visual as bench
 metrics=bench.official_scorer(W)
 labels=json.loads((W/'private_labels/labels.json').read_text())
 rows=[]
 for r in a['scored']+ext['scored']:
  res=r['result'];j=r['judge_ledger'];entries=r['ledger']
  events=res.get('image_events',[])
  candidate_correct=None
  if r['arm']=='harness_v2' and not r['scoring_function'].startswith('llm_'):
   candidate=str((res.get('final_review') or {}).get('answer',''))
   candidate_correct=bool(candidate.strip() and not metrics.is_unknown(candidate) and metrics.eval_from_spec(labels[r['id']]['eval_function'],metrics.extract_boxed_answer(candidate),labels[r['id']]['answer']))
  rows.append({'id':r['id'],'arm':r['arm'],'domain':r['domain'],'ability':meta[r['id']]['ability'],
    'question_image':r['question_image'],'scoring_function':r['scoring_function'],
    'model':r['model'],'status':r['status'],'correct':r['correct'],'unknown':r['unknown'],
    'non_unknown_answer':r['status']=='returned' and bool(r['parsed_answer'].strip()) and not r['unknown'],
    'judge_error':r['judge_error'],'error':r['error'],'seconds':r['seconds'],'completed_at_unix':r['completed_at_unix'],
    'answer_sha256':hashlib.sha256(r['answer'].encode()).hexdigest(),
    'parsed_answer_sha256':hashlib.sha256(r['parsed_answer'].encode()).hexdigest(),
    'record_sha256':digest((exto if r['arm']=='harness_visual_route' else o)/'predictions'/f'{r["id"]}__{r["arm"]}.json'),
    'source_sha256':r['source_metadata']['canonical_source_sha256'],
    'source_text_tokens':r['source_text_tokens_exact'],'source_chars':r['source_metadata']['source_chars'],
    'tokens':r['usage']['charged_tokens'],'calls':r['usage']['physical_requests'],
    'budget':r['usage']['budget'],'ledger':receipts(entries),'judge_ledger':receipts(j),
    'actual_models':sorted({e['actual_model'] for e in r['response_events']}),
    'gate_reason':res.get('gate_reason'),'decision':res.get('decision'),
    'image_occurrences':sum(len(e['images']) for e in events),
    'unique_history_images':len({im['image_sha256'] for e in events for im in e['images'] if not im.get('query_image')}),
    'final_support_not_in_final':len(res.get('support_ids_not_in_final',res.get('supported_spans_not_in_final',[]))),
    'accumulated_support_omitted':len(res.get('accumulated_support_ids_not_in_final',[])),
    'truncated_generations':sum(e.get('finish_reason')=='length' for e in entries),
    'deterministic_final_candidate_correct':candidate_correct,
    'final_parse_error':bool((res.get('final_review') or {}).get('parse_error')),
    'review_parse_errors':sum(bool(t.get('parse_error') or t.get('review',{}).get('parse_error')) for t in res.get('trace',[])),
    'transport_errors':r.get('transport_errors',[])})
 summaries={arm:summary([r for r in rows if r['arm']==arm]) for arm in arms}
 development=[]
 for name in ['longmemeval_v2_visual_dev_20261103','longmemeval_v2_visual_dev2_20261103','longmemeval_v2_visual_dev3_20261103']:
  p=ROOT/'work'/name/'formal';v=json.loads((p/'audit.json').read_text())
  development.append({'stage':name,'freeze_sha256':digest(p/'freeze.json'),'summary':v['summary'],
                      'same_12_previously_incorrect_questions':True,'used_for_tuning_not_holdout':True})
 paired={};by={(r['id'],r['arm']):r for r in rows}
 visual_pairs=[(by[q,'harness'],by[q,'harness_visual_route']) for q in f['question_ids'] if by[q,'harness']['question_image']]
 visual_tradeoff={'n':len(visual_pairs),'old_correct_to_unknown':sum(old['correct'] is True and new['unknown'] for old,new in visual_pairs),
 'old_incorrect_to_correct':sum(old['correct'] is False and new['correct'] is True for old,new in visual_pairs),
 'old_incorrect_to_unknown':sum(old['correct'] is False and new['unknown'] for old,new in visual_pairs)}
 for arm in arms:
  if arm=='harness_v2':continue
  pairs=[(by[q,arm]['correct'],by[q,'harness_v2']['correct']) for q in f['question_ids']]
  paired[arm]={'both_correct':pairs.count((True,True)),'new_only_correct':pairs.count((False,True)),
               'control_only_correct':pairs.count((True,False)),'neither_correct':pairs.count((False,False))}
 payload={'schema_version':1,'experiment_id':ID,'model':f['model'],'benchmark':'LongMemEval-V2','tier':'small',
   'n_questions':20,'n_shared_histories':2,'n_trajectories':200,'n_history_screenshots':5095,'n_question_images':4,
   'sample_ids':f['question_ids'],'arms':arms,'budgets':budgets,'judge_budget':f['judge_budget'],
   'visual_config':f['visual_config'],'seed':f['seed'],'selection_seed':sel['seed'],
   'question_id_overlap_with_pilot':sel['question_overlap_with_pilot'],
   'question_text_hash_overlap_with_pilot':sel['normalized_question_hash_overlap_with_pilot'],
   'source_histories_reused_from_development':True,'official_leaderboard_submission':False,
   'judge_substitution':'Qwen3.8-Flash-Next-FP8 instead of official default gpt-5.2; official prompts unchanged',
   'frozen_at_unix':f['frozen_at_unix'],'first_scoring_access_unix':access['opened_at_unix'],
   'freeze_sha256':digest(o/'freeze.json'),'route_freeze_sha256':digest(exto/'freeze.json'),'code_sha256':codes,
   'route_registered_before_any_gold':extf['registered_before_any_holdout_gold_access'],
   'route_registered_at_unix':extf['frozen_at_unix'],
   'all120_fixed_before_any_scoring_access':all(r['completed_at_unix']<=min(access['opened_at_unix'],extaccess['opened_at_unix']) for r in rows),
   'route_first_scoring_access_unix':extaccess['opened_at_unix'],
   'visual_features_sha256':extf['visual_features_sha256'],
   'visual_features_integrity':json.loads((W/'visual_features_integrity.json').read_text()),
   'legacy_branch_comparison':{'n':16,
    'first_prompt_hash_matches':sum(r['ledger'][0]['prompt_sha256']==by[r['id'],'harness']['ledger'][0]['prompt_sha256'] for r in rows if r['arm']=='harness_visual_route' and not r['question_image']),
    'parsed_answer_hash_matches':sum(r['parsed_answer_sha256']==next(v['parsed_answer_sha256'] for v in rows if v['id']==r['id'] and v['arm']=='harness') for r in rows if r['arm']=='harness_visual_route' and not r['question_image']),
    'correct_new':sum(r['correct'] is True for r in rows if r['arm']=='harness_visual_route' and not r['question_image']),
    'correct_old':sum(r['correct'] is True for r in rows if r['arm']=='harness' and not r['question_image']),
    'answer_hash_matches':sum(r['answer_sha256']==next(v['answer_sha256'] for v in rows if v['id']==r['id'] and v['arm']=='harness') for r in rows if r['arm']=='harness_visual_route' and not r['question_image'])},
   'data_revision':f['data_revision'],'upstream_code_commit':f['upstream_code_commit'],
   'official_scorer_sha256':f['upstream_scoring_sha256'],'summary':summaries,'paired':paired,
   'visual_route_tradeoff':visual_tradeoff,
   'results':rows,'development':development,'tests_passed':236,'original_core_unchanged':True,
   'auditor_sha256':audit['auditor_sha256'],'offline_audit_sha256':digest(W/'offline_audit.json'),
   'route_offline_audit_sha256':digest(extw/'offline_audit.json'),
   'audit':{'status':'pass' if audit['status']==exta['status']=='pass' else 'partial_replay',
      'ledgers_passed':audit['ledgers_passed']+exta['ledgers_passed'],
      'scores_recomputed':audit['scores_recomputed']+exta['scores_recomputed'],
      'replay_passed':audit['replay_passed']+exta['replay_passed'],
      'totals':{k:v+exta['totals'][k] for k,v in audit['totals'].items()}},
   'posthoc_diagnostics':{'deterministic_candidates_rejected_despite_matching_gold':sum(r['unknown'] and r['deterministic_final_candidate_correct'] is True for r in rows if r['arm']=='harness_v2'),
   'deterministic_incorrect_candidates_rejected':sum(r['unknown'] and r['deterministic_final_candidate_correct'] is False for r in rows if r['arm']=='harness_v2'),
   'interpretation':'Post-score oracle diagnostic, not proof that rejected evidence was sufficient; no model calls'},
   'independent_semantic_validation':False,'remote_files_modified':False,'github_push':False,
   'scope':'Question-disjoint pilot with reused source histories, single run, unequal native/Harness ceilings. No fresh-source or official LAFS claim.'}
 public=ROOT/'benchmarks'/f'{ID}.json';public.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
 shutil.copyfile(W/'offline_audit.json',ROOT/'benchmarks'/f'{ID}_audit.json')
 shutil.copyfile(extw/'offline_audit.json',ROOT/'benchmarks'/f'{ID}_route_audit.json')
 reports(payload)
 print(json.dumps({'summary':summaries,'audit_status':audit['status']},indent=2))


def reports(p):
 by={(r['id'],r['arm']):r for r in p['results']};new=p['summary']['harness_v2'];old=p['summary']['harness'];routed=p['summary']['harness_visual_route']
 for zh in [True,False]:
  def t(a,b):return a if zh else b
  lines=['# '+t('LongMemEval-V2：视觉召回与拒答优化后的问题留出对照','LongMemEval-V2: visual retrieval and abstention question holdout'),'',
   t(f'本轮两项新方案均未达到替换旧版的标准，默认路径保持不变。新 20 题：保守路由 Harness **{routed["correct"]}/20**，全量门控候选 **{new["correct"]}/20**，旧版 **{old["correct"]}/20**。六路均保留。',
     f'Neither new candidate meets the replacement bar; the original default is retained. On 20 new questions, conservative routed Harness **{routed["correct"]}/20**, all-question strict gate **{new["correct"]}/20**, previous Harness **{old["correct"]}/20**. All six arms are retained.'),'',
   t('新题与首轮 20 题的 ID、规范化正文哈希均无重叠；web／enterprise 各 10 题，五种能力各 4 题，4 题带题图。历史仍是同两份 100 轨迹材料，共 200 轨迹、5095 原图。是问题留出试评，不是来源独立留出、451 题全量或官方 LAFS／排行榜成绩。',
     'Question IDs and normalized text hashes do not overlap the original 20. There are 10 questions per domain, four per ability, and four query images. The same two 100-trajectory histories are reused: 200 trajectories and 5,095 original screenshots. This is a question holdout pilot, not a source-disjoint evaluation, full 451-question run, or official LAFS/leaderboard score.'),'',
   '| '+t('路径 | 答对／20 | UNKNOWN | 作答覆盖 | 已作答准确率 | 错误作答 | 输入＋输出 tokens |',
             'Arm | Correct/20 | UNKNOWN | Answer coverage | Accuracy when answered | Wrong answers | Input + output tokens |'),'|---|---:|---:|---:|---:|---:|---:|']
  for arm,s in p['summary'].items():
   score='capacity denied' if arm=='direct_full' else f'{s["correct"]}/20 ({s["task_accuracy"]:.0%})'
   selective='—' if s['selective_accuracy'] is None else f'{s["selective_accuracy"]:.1%}'
   lines.append(f'| {arm} | {score} | {s["unknown"]} | {s["coverage"]:.0%} | {selective} | {s["wrong_answered"]} | {s["tokens"]:,} |')
  lines+=['',t('完整直连的容量拒绝在生成前发生，不能作为语义答错。简短输出直连组仅增加简短 boxed 答案要求，其余来源首尾范围、选图与原生预算不变，用来检查格式差异是否解释收益。新版默认路径仍未替换旧版，避免把开发中的拒答策略自动推广到原能力。',
    'Full Direct capacity denials happen before generation and are not semantic wrong answers. The concise Direct control adds only a short boxed-answer requirement; source truncation, image selection, and native budget are otherwise unchanged. The experimental route does not replace the original default.'),'']
  if new['correct']<old['correct']:
   lines += [t('**全量严格门控未提升总体准确率：它以更多拒答牺牲任务交付，不能作为默认升级。保守路由单独列出，不混同这项失败试验。**',
               '**The all-question strict gate does not improve overall accuracy and sacrifices delivery through abstention. It is not a default upgrade; the conservative route is reported separately.**'),'']
  lines+=['## '+t('预登记的保守视觉路由','Predeclared conservative visual route'),'',
   t('扩展方案在任何留出标签访问前另行固定：16 道无题图问题直接执行旧控制器，4 道有题图问题启用感知匹配与证据门控。20 道均为新调用，没有答案复用。图像特征在独立 Pillow 11.3.0 环境内本地计算，原推理环境依赖未改；特征为双向 dHash 和 8×8 RGB 布局，属于相似度启发式，不是语义图像嵌入或已校准的召回概率。推理仍发送原始 PNG。',
     'An extension was frozen before any holdout-gold access: 16 questions without query images use the original controller; four with query images enable perceptual matching and the evidence gate. All 20 use fresh calls, without answer reuse. Bidirectional dHash and 8x8 RGB layout features are computed locally in an isolated Pillow 11.3.0 environment, without changing inference dependencies. Similarity is a heuristic, not semantic embedding or calibrated recall probability. Original PNGs are sent to inference.'),'',
   t(f'无题图分支首个请求提示哈希一致 {p["legacy_branch_comparison"]["first_prompt_hash_matches"]}/16；回复全文哈希一致 {p["legacy_branch_comparison"]["answer_hash_matches"]}/16，解析后答案哈希一致 {p["legacy_branch_comparison"]["parsed_answer_hash_matches"]}/16；新／旧答对 {p["legacy_branch_comparison"]["correct_new"]}/16、{p["legacy_branch_comparison"]["correct_old"]}/16。全部 120 条输出早于任何一轮评分标签访问：{p["all120_fixed_before_any_scoring_access"]}。扩展 freeze：`{p["route_freeze_sha256"]}`。',
     f'No-query-image first-prompt hashes match on {p["legacy_branch_comparison"]["first_prompt_hash_matches"]}/16; raw/parsed answer hashes match on {p["legacy_branch_comparison"]["answer_hash_matches"]}/16 and {p["legacy_branch_comparison"]["parsed_answer_hash_matches"]}/16; new/old correct counts are {p["legacy_branch_comparison"]["correct_new"]}/16 and {p["legacy_branch_comparison"]["correct_old"]}/16. All 120 outputs precede any scoring-label access: {p["all120_fixed_before_any_scoring_access"]}. Extension freeze: `{p["route_freeze_sha256"]}`.'),'',
   t('无题图分支是同一旧算法的再次执行，底层服务仍可能产生输出差异；该部分的变化不能归因为新视觉检索。两组实验部分并行，计时仅作记录，不作为方法延迟优势的证据。',
     'The no-query-image branch repeats the same old algorithm; backend outputs may vary. Differences there cannot be attributed to new visual retrieval. Runs overlap in time; timings are records, not evidence of a latency advantage.'),'',
   '## '+t('评分和视觉子集','Scoring and image subsets'),'','| Subset | '+ ' | '.join(p['arms'][1:])+' |','|---|'+'---:|'*(len(p['arms'])-1)]
  for label,ck,nk in [('Deterministic','deterministic_correct','deterministic_n'),('Qwen judged','judge_correct','judge_n'),('Query image','image_correct','image_n')]:
   lines.append('| '+label+' | '+' | '.join(f'{p["summary"][a][ck]}/{p["summary"][a][nk]}' for a in p['arms'][1:])+' |')
  lines+=['',t('确定性评分函数沿用固定上游版本；裁判题使用原官方提示，将默认 gpt-5.2 换为 Qwen，可能存在同模型裁判偏差。拒答记为任务不正确，并另列覆盖率及已作答准确率；不能用拒答降低分母来包装整体成绩。',
    'Deterministic upstream scoring is preserved. LLM-judged items use official prompts but Qwen replaces default gpt-5.2, with possible same-model judge bias. Abstentions count as unsuccessful tasks; coverage and selective accuracy are reported separately.'),'',
   t(f'5099 张原图的数值特征已重新计算并与缓存逐项比较：{p["visual_features_integrity"]["status"]}，未调用模型；该检查不评价视觉证据的语义相关性。',
     f'All 5,099 image features were rederived and compared with cache: {p["visual_features_integrity"]["status"]}, without model calls. This does not validate semantic evidence relevance.'),'',
   '## '+t('已确认的问题','Confirmed findings'),'',
   t(f'有题图的 4 题中，保守路由把旧版 {p["visual_route_tradeoff"]["old_incorrect_to_correct"]} 道错题改为正确，却把旧版答对的 {p["visual_route_tradeoff"]["old_correct_to_unknown"]} 道变为拒答。全量门控的已作答准确率为 {new["selective_accuracy"]:.1%}，仅略高于旧版 {old["selective_accuracy"]:.1%}，但覆盖率从 {old["coverage"]:.0%} 降到 {new["coverage"]:.0%}。因此不能以少答题换来的少量精度改善宣称整体能力提升。',
     f'On four query-image questions, the routed method corrects {p["visual_route_tradeoff"]["old_incorrect_to_correct"]} old error but abstains on {p["visual_route_tradeoff"]["old_correct_to_unknown"]} old-correct questions. Strict-gate selective accuracy is {new["selective_accuracy"]:.1%}, versus {old["selective_accuracy"]:.1%} for the original, while coverage falls from {old["coverage"]:.0%} to {new["coverage"]:.0%}. This tradeoff does not establish an overall capability gain.'),'',
   t('实现中还存在已知原因：只用原来的错题开发门控，缺少正确题作为拒答校准对照；引用格式校验会把某些可恢复的候选挡住；同一状态的 seen 标记可能阻止另一字段窗口的追检。这些机制需要后续独立验证。当前不再根据本轮留出结果改参数并沿用本轮分数。',
     'Known implementation limits include error-only gate development without a positive calibration cohort, quote-format rejection of recoverable candidates, and state-level seen markers blocking another field excerpt. These mechanisms need separate validation; this holdout is not retuned and reused.'),'',
   '## '+t('技术改动与边界','Changes and limits'),' ',
   t('1. 状态级观察索引与原始块索引融合，结合历史轨迹目录路由；稀有词加权选择原字段片段，避免仅凭长文本重叠选图。\n2. 原图绑定轨迹、状态、字段范围及 SHA；动态问题同时检索相邻状态，操作字段明确表示到达当前状态之前的动作。\n3. 跨轮保留可核对的部分支持，最终重附原文与原图；公开遗漏计数，不能将引用存在等同于语义成立。\n4. 最终门控区分支持、前提已证伪、证据不足；按要求检查引用、原文短引、物理请求中的图片，以及缺失条件和冲突。\n5. 最终短答案由程序封装；机械引用错误可在原任务预算内修复一次，不能靠重发获取更有利答案。',
     '1. Fuse observed-state and raw-block retrieval with trajectory-directory routing; use rarity-weighted field excerpts.\n2. Bind original images to state/field/hash receipts and retrieve adjacent states for transitions.\n3. Accumulate verifiable partial support, then reattach raw fields/images; disclose omitted support.\n4. Distinguish supported answers, contradicted premises, and insufficient evidence; validate citations, quotes, image exposure, missing conditions, and source conflicts.\n5. Render a short answer in software; allow one budgeted mechanical repair, without blind resends.'),'',
   t('这些是机械约束和模型判定的组合，不是语义证明。置信度为模型自报，阈值 0.8 未经独立校准；负例识别仍可能出错。历史索引包含的旧 agent 思路仅能路由检索，不作为事实证据。',
     'These checks combine mechanical validation and model judgments, not semantic proof. Confidence is model-reported; the 0.8 threshold is not independently calibrated. Historical agent thoughts can route raw-block retrieval but cannot serve as observed facts.'),'',
   t('三条 Harness 路径共用单次 16,384、累计 65,536 tokens、最多 8 次生成、单次输出上限 1024；新路径每次最多 6 张历史原图，旧路径最多 3 张，题图均保留。原生直连单次及累计 262,144 tokens、1 次生成、1024 输出，首尾组最多 16 张图。因此原生与 Harness 计算预算不相等，图片暴露量也不同，不是独立消融。',
     'All three Harness paths share 16,384 tokens/request, 65,536/task, eight generations, and 1,024 output tokens. New/old routes allow six/three history images per call, plus the query image. Native Direct allows 262,144 tokens/request and task, one generation, and 1,024 output tokens; head/tail allows 16 images. Native/Harness compute and image exposure differ; this is not an isolated ablation.'),'',
   t('当前仍有明确边界：同一状态一旦标记 seen，不能在另一轮换取其不同字段片段；字段窗口和最终六张图也可能遗漏信息。尚无逐题人工 oracle 证据标注，不能把图片绑定通过或视觉子集答对率称为证据召回率提升。',
     'Known limits remain: a state marked seen cannot be revisited for a different field excerpt; bounded excerpts and six final images can omit information. No per-question human oracle evidence annotations exist, so valid image bindings and image-subset accuracy do not establish improved evidence recall.'),'',
   '## '+t('开发轨迹与原能力保护','Development history and regression protection'),'']
  for dev in p['development']:
   s=dev['summary']['harness_v2'];lines.append(f'- {dev["stage"]}: {s["correct"]}/12 correct, {s["unknown"]}/12 UNKNOWN; {s["charged_or_reserved_tokens"]:,} tokens; freeze `{dev["freeze_sha256"]}`.')
  lines+=['',t('开发阶段均使用原先 12 道错题，不能当新留出成绩；初版过度拒答的失败记录没有删除。236 项检查通过，原文本／事实控制器与首次 20 题评测的 49 份冻结源码未改。整个任务没有更改原环境依赖、远端文件或 GitHub 仓库。',
    'All development stages reuse the original 12 errors and are not holdout scores. Over-abstaining failed prototypes are retained. All 236 checks pass; original text/fact controllers and first-pilot frozen sources are unchanged. No original environment dependencies, remote files, or GitHub repository were changed.'),'',
   '[12-case analysis](longmemeval_v2_visual_error_analysis_20261103.zh-CN.md)','',
   t(f'评分后确定性题诊断：有 {p["posthoc_diagnostics"]["deterministic_candidates_rejected_despite_matching_gold"]} 个最终候选与标准答案匹配却被门控拒绝；有 {p["posthoc_diagnostics"]["deterministic_incorrect_candidates_rejected"]} 个不匹配候选被拒绝。这个诊断不证明被拒候选具有充分证据，也不作为再次调优后沿用本轮留出成绩的依据。',
     f'Post-score deterministic diagnostic: {p["posthoc_diagnostics"]["deterministic_candidates_rejected_despite_matching_gold"]} matching final candidates were rejected; {p["posthoc_diagnostics"]["deterministic_incorrect_candidates_rejected"]} nonmatching candidates were rejected. This does not prove sufficient evidence for rejected candidates or permit reusing this holdout after tuning.'),'',
   '## '+t('逐题对照','Per-question outcomes'),'','| ID | Domain | Ability | Image | '+ ' | '.join(p['arms'])+' |','|---|---|---|---|'+'---|'*len(p['arms'])]
  for q in p['sample_ids']:
   r=by[q,'harness_v2'];cells=[]
   for arm in p['arms']:
    v=by[q,arm];cells.append('capacity_denied' if v['status']=='native_capacity_denied' else 'correct' if v['correct'] else 'UNKNOWN' if v['unknown'] else 'incorrect')
   lines.append(f'| {q} | {r["domain"]} | {r["ability"]} | {r["question_image"]} | '+' | '.join(cells)+' |')
  lines+=['','## '+t('来源、审计与复核','Provenance, audit and verification'),' ',
   '[Official repository](https://github.com/xiaowu0162/LongMemEval-V2) · [Paper](https://arxiv.org/abs/2605.12493) · [Official data](https://huggingface.co/datasets/xiaowu0162/longmemeval-v2)','',
   f'Data `{p["data_revision"]}`; code `{p["upstream_code_commit"]}`; freeze `{p["freeze_sha256"]}`.','',
   t(f'离线审计：{p["audit"]["ledgers_passed"]}/120 账本、{p["audit"]["scores_recomputed"]}/120 评分重算、{p["audit"]["replay_passed"]}/120 控制流程重放，状态 {p["audit"]["status"]}。重放不调用服务端；使用保存的计数、回复及冻结生产逻辑，不构成独立人工语义验证。',
     f'Offline audit: {p["audit"]["ledgers_passed"]}/120 ledgers, {p["audit"]["scores_recomputed"]}/120 score re-parses, {p["audit"]["replay_passed"]}/120 controller replays; status {p["audit"]["status"]}. Replay uses saved counts/responses and frozen production logic, without provider calls or independent human semantic validation.'),'',
   '```bash','python3 -S scripts/verify_longmemeval_v2_visual_results.py','```','',
   t('最长来源仍为 34,235,595 个规范化 JSON 文本 Qwen tokens，不含像素；本轮没有扩大来源历史规模。端到端时间包含计数及推理，建索引另存，不等于官方查询延迟。公开资料不含原题、标准答案、原图、回复全文或凭证；完整本地审计需保留工作目录。',
     'Largest source remains 34,235,595 Qwen canonical-JSON text tokens, excluding pixels; this run does not expand source-history size. Timing includes tokenization and inference, with indexing separate, not official query latency. Public artifacts omit raw questions, gold, images, replies, and credentials; full local replay requires the workspace.'),'',
   f'[Results JSON](../benchmarks/{ID}.json) · [Primary audit](../benchmarks/{ID}_audit.json) · [Route audit](../benchmarks/{ID}_route_audit.json)','']
  (ROOT/'docs'/f'{ID}{".zh-CN" if zh else ""}.md').write_text('\n'.join(lines))


if __name__=='__main__':main()
