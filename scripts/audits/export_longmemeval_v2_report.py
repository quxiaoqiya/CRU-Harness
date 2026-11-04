"""Whitelist publication metadata after all inference/scoring/offline audits."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import statistics

ROOT=Path(__file__).resolve().parents[2]
WORK=ROOT/'work/longmemeval_v2_qwen_20261103'
ID='longmemeval_v2_qwen_20261103'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def usage(entries):
    known=[e for e in entries if e.get('usage_known')]
    return {'physical_generations':len(entries),
        'known_input_tokens':sum(e['usage']['prompt_tokens'] for e in known),
        'known_output_tokens':sum(e['usage']['completion_tokens'] for e in known),
        'charged_or_reserved_tokens':sum(e['charged_tokens'] for e in entries),
        'unknown_usage_requests':sum(not e.get('usage_known') for e in entries)}


def receipts(entries):
    fields = ('physical_request', 'prompt_sha256', 'admission_prompt_tokens',
              'requested_output_tokens', 'charged_tokens', 'status', 'usage_known', 'finish_reason')
    return [{**{k: e.get(k) for k in fields},
             'known_prompt_tokens': e['usage']['prompt_tokens'] if e.get('usage_known') else None,
             'known_completion_tokens': e['usage']['completion_tokens'] if e.get('usage_known') else None}
            for e in entries]


def main():
    out=WORK/'formal'
    assert len(list((out/'predictions').glob('*.json')))==60
    assert (out/'audit.json').exists() and (out/'scoring_access.json').exists() and (WORK/'offline_audit.json').exists()
    frozen=json.loads((out/'freeze.json').read_text())
    scored=json.loads((out/'audit.json').read_text())
    audit=json.loads((WORK/'offline_audit.json').read_text())
    assert audit['ledgers_passed']==60 and audit['scores_recomputed']==60
    selection=json.loads((WORK/'selection_manifest.json').read_text())
    metadata={r['id']:r for r in selection['selected']}
    rows=[]
    for row in scored['scored']:
        image_events=row['result'].get('image_events',[])
        rows.append({'id':row['id'],'domain':row['domain'],'ability':metadata[row['id']]['ability'],
            'question_type':row['question_type'],'question_image':row['question_image'],
            'arm':row['arm'],'model':row['model'],'status':row['status'],
            'correct':row['correct'],'unknown':row['unknown'],'judge_error':row['judge_error'],
            'scoring_function':row['scoring_function'],
            'answer_sha256':hashlib.sha256(row['answer'].encode()).hexdigest(),
            'source_sha256':row['source_metadata']['canonical_source_sha256'],
            'source_text_tokens_exact':row['source_text_tokens_exact'],
            'source_chars':row['source_metadata']['source_chars'],
            'trajectory_count':row['source_metadata']['trajectory_count'],
            'source_state_count':row['source_metadata']['state_count'],
            'source_trajectory_sha256':row['source_metadata']['source_trajectory_sha256'],
            'record_sha256':digest(out/'predictions'/f'{row["id"]}__{row["arm"]}.json'),
            'budget':row['usage']['budget'],'usage':usage(row['ledger']),
            'seconds':row['seconds'],'completed_at_unix':row['completed_at_unix'],
            'actual_models':sorted({e['actual_model'] for e in row['response_events']}),
            'transport_errors':row.get('transport_errors',[]),'terminal_error':row['error'],
            'image_occurrences':sum(len(event['images']) for event in image_events),
            'image_sha256':sorted({im['image_sha256'] for e in image_events for im in e['images']}),
            'final_source_spans':row['result'].get('final_spans',[]),
            'retained_source_spans':row['result'].get('retained_source_spans'),
            'supported_spans_not_in_final_count':len(row['result'].get('supported_spans_not_in_final',[])),
            'review_parse_errors':sum(bool(t.get('parse_error')) for t in row['result'].get('trace',[])),
            'judge_usage':usage(row['judge_ledger']),
            'ledger_receipts':receipts(row['ledger']), 'judge_receipts':receipts(row['judge_ledger'])})
    summary={}
    for arm in frozen['arms']:
        group=[r for r in rows if r['arm']==arm]
        total_usage={k:sum(r['usage'][k] for r in group) for k in usage([])}
        summary[arm]={'n':len(group),'correct':sum(r['correct'] is True for r in group),
            'task_success_rate':sum(r['correct'] is True for r in group)/len(group),
            'returned':sum(r['status']=='returned' for r in group),
            'semantic_accuracy_among_returned': (sum(r['correct'] is True for r in group)/sum(r['status']=='returned' for r in group)) if any(r['status']=='returned' for r in group) else None,
            'max_correct_source_text_tokens':max((r['source_text_tokens_exact'] for r in group if r['correct'] is True),default=None),
            'domain':{d:{'n':sum(r['domain']==d for r in group),'correct':sum(r['correct'] is True and r['domain']==d for r in group)} for d in ('web','enterprise')},
            'unknown':sum(r['unknown'] for r in group),
            'judge_failures':sum(r['judge_error'] is not None for r in group),
            'status_counts':dict(Counter(r['status'] for r in group)),
            'mean_seconds':statistics.mean(r['seconds'] for r in group),
            'median_seconds':statistics.median(r['seconds'] for r in group),
            'mean_charged_or_reserved_tokens':total_usage['charged_or_reserved_tokens']/len(group),
            'usage':total_usage,
            'deterministic_correct':sum(r['correct'] is True for r in group if not r['scoring_function'].startswith('llm_')),
            'deterministic_n':sum(not r['scoring_function'].startswith('llm_') for r in group),
            'llm_judged_correct':sum(r['correct'] is True for r in group if r['scoring_function'].startswith('llm_')),
            'llm_judged_n':sum(r['scoring_function'].startswith('llm_') for r in group),
            'question_image_correct':sum(r['correct'] is True for r in group if r['question_image']),
            'question_image_n':sum(r['question_image'] for r in group),
            'ability':{ability:{'n':sum(r['ability']==ability for r in group),
                'correct':sum(r['correct'] is True and r['ability']==ability for r in group)}
                for ability in sorted({r['ability'] for r in group})}}
    inputs=sorted({(r['source_sha256'],r['source_text_tokens_exact'],r['source_chars'],r['source_state_count']) for r in rows})
    access=json.loads((out/'scoring_access.json').read_text())
    preflight=[]
    for name in ['vision_canary_result.json','adapter_canary_result.json','controller_smoke_result.json']:
        value=json.loads((WORK/name).read_text());preflight.extend(value.get('ledger',[]))
    payload={'schema_version':1,'experiment_id':ID,'benchmark':'LongMemEval-V2','tier':'small',
        'dataset':'xiaowu0162/longmemeval-v2','data_revision':frozen['data_revision'],
        'official_code_commit':frozen['upstream_code_commit'],
        'official_scorer_sha256':frozen['upstream_scoring_sha256'],
        'freeze_sha256':digest(out/'freeze.json'),'frozen_at_unix':frozen['frozen_at_unix'],
        'first_scored_at_unix':access['opened_at_unix'],'all60_fixed_before_scoring_label_access':True,
        'model':frozen['model'],'n_questions':20,'n_unique_haystacks':2,'n_unique_trajectories':200,
        'n_history_screenshots':5095,'n_question_screenshots':4,'n_model_arm_results':60,
        'selection_seed':selection['seed'],'selection_rule':selection['selection'],
        'sample_ids':[r['id'] for r in selection['selected']],
        'judge_budget':frozen['judge_budget'],'budgets':frozen['budgets'],'trajectory_config':frozen['trajectory_config'],
        'wire':{'temperature':0,'seed':20261103,'enable_thinking':False,'preserve_thinking':False,'implicit_retries':0},
        'summary':summary,'results':rows,
        'source_text_measurements':[{'source_sha256':sha,'canonical_text_tokens':tokens,'source_chars':chars,
             'state_count':states,'pixels_are_excluded':True} for sha,tokens,chars,states in inputs],
        'code_sha256':frozen['code_sha256'],
        'auditor_sha256':audit['auditor_sha256'],'offline_audit_sha256':digest(WORK/'offline_audit.json'),
        'audit_summary':{k:audit[k] for k in ['status','ledgers_passed','scores_recomputed','trajectory_passed',
            'trajectory_unavailable_or_mismatched','totals','provider_or_tokenizer_calls_made']},
        'preflight_usage':usage(preflight),
        'judge_substitution':{'official_default':'gpt-5.2','used':frozen['model'],
            'official_prompts_and_deterministic_functions_preserved':True},
        'official_leaderboard_submission':False,'official_LAFS_score_measured':False,
        'same_compute_comparison':False,'original_58_source_fingerprints_unchanged':True,
        'limitations':['20 balanced pilot questions, not the full 451; only 2 shared source histories',
            'Canonical JSON text counts exclude screenshot pixels and differ from paper tokenizer units',
            'Full Direct capacity denials are not semantic wrong answers',
            'Native head/tail is lossy and has a larger request/total budget than Harness',
            'Qwen judge substitutes for the official default; no leaderboard/LAFS claim',
            'End-to-end time includes counting and generation; offline indexing is separate',
            'Replay reuses production implementation and recorded tokenizer counts; it does not prove semantic entailment']}
    by={(r['id'],r['arm']):r for r in rows}
    pairs=[(by[q,'direct_headtail_native']['correct'],by[q,'harness']['correct']) for q in payload['sample_ids']]
    payload['paired_native_vs_harness']={
        'both_correct':pairs.count((True,True)),
        'harness_only_correct':pairs.count((False,True)),
        'native_only_correct':pairs.count((True,False)),
        'neither_correct':pairs.count((False,False))}
    payload['judge_usage']={k:sum(r['judge_usage'][k] for r in rows) for k in usage([])}
    payload['quality_diagnostics']={
        'harness_incorrect':sum(r['correct'] is False for r in rows if r['arm']=='harness'),
        'harness_review_parse_errors':sum(r['review_parse_errors'] for r in rows if r['arm']=='harness'),
        'harness_supported_spans_not_in_final':sum(r['supported_spans_not_in_final_count'] for r in rows if r['arm']=='harness'),
        'semantic_coverage_is_not_proved_by_retaining_reported_supported_spans':True}
    public=ROOT/'benchmarks'/f'{ID}.json'
    public.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    write_reports(payload, audit)
    print(json.dumps({'public_json':str(public.relative_to(ROOT)),'summary':summary},indent=2))


def write_reports(p, audit):
    summaries=p['summary'];native=summaries['direct_headtail_native'];h=summaries['harness']
    ratio=h['usage']['charged_or_reserved_tokens']/native['usage']['charged_or_reserved_tokens']
    for zh in (True,False):
        def t(cn,en):return cn if zh else en
        lines=['# '+t('LongMemEval-V2：Qwen 直连与多模态 CRU-Harness 对照',
                      'LongMemEval-V2: Qwen Direct versus multimodal CRU-Harness'),'',
            t('本轮在本地 Mac 完成：Harness **8/20（40%）**，可用的原生首尾直连 **1/20（5%）**，提高 **35 个百分点**。完整直连 20 题均在生成前因容量不足被本地准入拒绝，语义准确率不适用。仅调用已有 Qwen 推理服务，没有修改远端部署文件。',
              'Completed locally on Mac: Harness **8/20 (40%)**, usable native head/tail Direct **1/20 (5%)**, a **35 percentage point** gain. Full Direct was denied locally before generation on all 20 questions because of capacity; its semantic accuracy is undefined. Only the existing Qwen inference service was called; no remote deployment files were changed.'),'',
            t('这是预先固定的 **20 题 small 层级试评**：web 与 enterprise 各 10 题，五类能力各 4 题，4 题带题图。各领域共享一份 100 轨迹历史，共 200 条轨迹、5,095 张历史截图。不是 20 份独立历史，也不是完整 451 题或官方排行榜成绩。',
              'This is a frozen **20-question small-tier pilot**: 10 questions per domain, four per ability, and four with query images. Each domain shares one 100-trajectory history: 200 unique trajectories and 5,095 history screenshots in total. These are two shared histories, not 20 independent histories, the full 451-question evaluation, or official leaderboard scores.'),'',
            '| '+t('路径 | 正确／总题数 | 返回 | UNKNOWN | 平均输入＋输出 tokens | 平均端到端秒 |',
                      'Arm | Correct / planned | Returned | UNKNOWN | Mean input + output tokens | Mean end-to-end seconds |'),
            '|---|---:|---:|---:|---:|---:|']
        for arm,s in summaries.items():
            outcome='— (20 capacity denials)' if arm=='direct_full' else f'{s["correct"]}/20 ({s["task_success_rate"]:.0%})'
            lines.append(f'| {arm} | {outcome} | {s["returned"]} | {s["unknown"]} | {s["mean_charged_or_reserved_tokens"]:,.0f} | {s["mean_seconds"]:.2f} |')
        lines+=['',t(f'Harness 累计输入＋输出 **{h["usage"]["charged_or_reserved_tokens"]:,} tokens**，原生首尾直连 **{native["usage"]["charged_or_reserved_tokens"]:,} tokens**，减少 **{1-ratio:.2%}**。分别为 59 次与 20 次生成；另有 14 次评分调用与 4 次预检生成，独立记账。没有未知用量、API 失败或预算越界。',
                    f'Harness consumed **{h["usage"]["charged_or_reserved_tokens"]:,} input + output tokens**, versus **{native["usage"]["charged_or_reserved_tokens"]:,}** for native head/tail, an **{1-ratio:.2%}** reduction. There were 59 and 20 generations respectively, plus 14 separately accounted judge calls and four preflight generations. No unknown usage, API failures, or budget violations occurred.'),'',
            '## '+t('准确率拆分与失败分析','Accuracy breakdown and failures'),'']
        lines+=['| '+t('子集 | 原生首尾直连 | Harness |','Subset | Native head/tail | Harness |'),'|---|---:|---:|']
        for label,k,n in [(t('确定性官方评分','Deterministic official scoring'),'deterministic_correct','deterministic_n'),
                          (t('Qwen 裁判评分','Qwen-judged scoring'),'llm_judged_correct','llm_judged_n'),
                          (t('题目自带截图','Question includes an image'),'question_image_correct','question_image_n')]:
            lines.append(f'| {label} | {native[k]}/{native[n]} | {h[k]}/{h[n]} |')
        for ability,v in h['ability'].items():
            lines.append(f'| {ability} | {native["ability"][ability]["correct"]}/4 | {v["correct"]}/4 |')
        for domain,v in h['domain'].items():lines.append(f'| {domain} | {native["domain"][domain]["correct"]}/10 | {v["correct"]}/10 |')
        lines+=['',t('逐题配对：两路都答对 1 题，只有 Harness 答对 7 题，只有首尾直连答对 0 题，两路都未答对 12 题。Harness 仍有 **12/20 答错且没有 UNKNOWN**，首尾直连为 11 道错误与 8 道 UNKNOWN；必须继续改善证据不足时的拒答校准。4 道带题图样本中，Harness 仅答对 1 道，尚不足以证明稳定的视觉记忆能力。',
                    'Paired outcomes: one correct on both arms, seven correct only with Harness, none correct only with native head/tail, and 12 incorrect on both. Harness still has **12/20 incorrect answers and no UNKNOWNs**; native head/tail has 11 incorrect answers and eight UNKNOWNs. Abstention calibration needs improvement. Harness answered only one of four query-image questions correctly; stable visual memory ability is not established.'),'',
            t('本轮 review 解析错误为 0，已登记的支持证据范围最终丢失计数为 0。这证明已声明证据的机械保留，不证明召回完整或证据语义正确。下一轮应将召回遗漏、状态时序判断、视觉定位与错误前提识别分别诊断；当前结果不再用于反复调优后冒充新留出测试。',
              'There were zero review parse errors and zero reported support spans omitted from final evidence. This checks mechanical retention of declared evidence, not retrieval completeness or semantic entailment. Follow-up work should separate missed retrieval, temporal state reasoning, visual grounding, and false-premise detection, then use a fresh holdout rather than relabeling tuned results as new tests.'),'',
            '## '+t('上下文容量与预算','Source extent and budgets'),'']
        for source in p['source_text_measurements']:
            lines.append(f'- **{source["canonical_text_tokens"]:,} Qwen canonical-text tokens**; {source["source_chars"]:,} characters; {source["state_count"]:,} states. **{source["canonical_text_tokens"]/262144:.2f}×** native 262,144; **{source["canonical_text_tokens"]/16384:.2f}×** working ceiling 16,384.')
        lines+=['',t(f'最大来源与最长答对来源均为 **{h["max_correct_source_text_tokens"]:,} 个规范化文本 tokens**。这是完整轨迹 JSON 文本经 Qwen 精确计数的长度，不含截图像素，不等于官方论文的 tokenizer 口径；不是模型单次阅读长度，也没有测得 Harness 的硬极限。完整文本与原始截图留在本地外部记忆，模型按检索窗口读取证据。',
                    f'The largest source and longest correctly answered source both contain **{h["max_correct_source_text_tokens"]:,} canonical-text tokens**. This is exact Qwen counting of complete trajectory JSON text, excluding screenshot pixels, and differs from paper-tokenizer units. It is neither a single model input nor a measured hard maximum. Complete text and original screenshots remain in local external memory; the model reads selected evidence windows.'),'',
            t('完整直连与首尾直连：单次及累计 262,144 tokens、1 次生成、最多 1,024 输出。首尾直连截取两段原始文本，最多 16 张原图，按保留状态均匀抽取；完整多模态提示经精确计数后缩减至可准入。Harness：单次 16,384、累计 65,536 tokens、最多 8 次物理请求、最多 3 轮证据 review 加最终回答；每次最多 3 张历史原图，保留题图。**计算预算不相等**；这不是同预算因果估计，也未与全源视觉扫描、其他检索策略做消融。',
              'Full Direct and head/tail Direct allow 262,144 tokens per request and task, one generation, and 1,024 output tokens. Head/tail retains two original text spans and uniformly samples up to 16 original screenshots from retained states; exact multimodal counting governs admission. Harness allows 16,384 tokens per request, 65,536 per task, and eight physical requests: up to three evidence reviews plus a final answer, with up to three history images per call and the query image preserved. **Compute ceilings differ**. This is not a same-budget causal estimate or an ablation against complete visual scanning and alternative retrieval.'),'',
            '## '+t('新增适配与原能力保护','Opt-in adaptation and regression protection'),' ',
            t('新增 `long_context_agent/trajectory_memory.py`：保留所有状态字段，建立轨迹／状态／字符范围绑定，将原始截图与 SHA-256 绑定；先组装图片再计数和推理。检索复用既有 SourceIndex 与证据缺口协议，并累积已验证范围；最终输入使用官方领域提示及原图，不将未经验证的临时答案作为事实。没有预生成截图描述、缩图或答案缓存。',
              'New opt-in module `long_context_agent/trajectory_memory.py` preserves state fields, binds trajectory/state/character spans to original screenshot SHA-256 hashes, and assembles images before token admission and inference. Retrieval reuses SourceIndex and the evidence-gap protocol while accumulating validated spans. Final input uses official domain prompts and original images, without treating unverified tentative answers as facts. No generated screenshot captions, resized images, or answer caches are used.'),'',
            t('原 `harness.py`、包入口及文本／结构化事实控制器未修改。新增 11 项检查连同 89 项回归检查共 **100 passed**。此前 LongBench 公开核验的 58 份源码指纹保持不变。没有改变原 Python 环境依赖；轨迹模块按需导入。',
              'The original harness.py, package entry point, text controllers, and structured-fact controllers are unchanged. Eleven new checks plus 89 regression checks give **100 passed**. All 58 prior LongBench source fingerprints remain unchanged. Dependencies in the original Python environment were not changed; trajectory support is imported explicitly.'),'',
            '## '+t('评分隔离、审计与复核','Scoring isolation, audit and verification'),' ',
            t('元数据分层哈希选样，seed=20261103，不用题目正文或答案选样。选样预处理仅将标签搬运至隔离目录，全部 60 条终态输出固定后才启用评分阶段的标签访问；推理子进程由 Mac 沙盒禁止读取标签目录、原始带答案问题文件及报告目录。使用固定官方评分函数和裁判提示，但 LLM 裁判由默认 gpt-5.2 改为 Qwen3.8-Flash-Next-FP8。故 8 道裁判题有同模型评判偏差风险，不能混称官方 LAFS 分数。',
              'Metadata-stratified hash selection uses seed 20261103 without question text or gold. Preprocessing only copied labels into an isolated directory. Scoring-stage label access began after all 60 terminal inference outputs were fixed; the Mac sandbox denied the inference worker access to labels, the original answered-question file, and reports. Official scoring functions and judge prompts are pinned, but Qwen3.8-Flash-Next-FP8 replaces default gpt-5.2 as LLM judge. The eight judged questions therefore carry same-model judgment bias; no official LAFS score is claimed.'),'',
            f'Offline audit: **{audit["ledgers_passed"]}/60 ledgers**, **{audit["scores_recomputed"]}/60 score re-parses**, **{audit["trajectory_passed"]}/60 controller replays**, **507 image occurrences**, **79 generations** and **624 tokenizer events**. No provider/tokenizer calls during replay.','',
            t('重放核对保存的精确计数与生产控制器，复核官方评分结果和图片绑定；它不是独立的人工语义验证。公开 JSON 仅含白名单元数据、逐调用预算收据与哈希，原题、标准答案、模型原始回复、原图和凭证不随报告发布。公开核验器能重算指标和预算、核对源码与审计哈希，不能从未公开内容独立判断答案语义。',
              'Replay uses recorded exact counts and the production controller, reparses official scores, and checks image bindings. It is not independent human semantic verification. Public JSON contains whitelisted metadata, per-call budget receipts, and hashes; raw questions, gold, model responses, images, and credentials are excluded. Public verification recomputes arithmetic/budgets and checks code/audit hashes, not unpublished answer semantics.'),'',
            '```bash','python3 -S scripts/verify_longmemeval_v2_results.py','```','',
            t('保留本机完整数据时，可运行 `python scripts/audits/audit_longmemeval_v2.py` 离线重放，不会再次生成。推理与评分入口为 `python -m scripts.benchmark_longmemeval_v2 {freeze,run,score}`；`run/score` 使用既有 provider 环境文件，冻结目录不会被覆盖。新机器复跑需从下述固定上游版本重新取得数据并生成同格式选样／图片清单，不是只下载报告即可重跑模型。',
              'With the complete local workspace, `python scripts/audits/audit_longmemeval_v2.py` replays offline without new generation. Inference/scoring entry: `python -m scripts.benchmark_longmemeval_v2 {freeze,run,score}`; run/score read an existing provider environment file and preserve the frozen directory. A fresh machine must obtain the pinned upstream data and prepare equivalent selection/image manifests; the report alone is insufficient to rerun inference.'),'',
            '## '+t('逐题结果','Per-question outcomes'),' ',
            '| ID | Domain | Ability | Query image | Full Direct | Native head/tail | Harness |',
            '|---|---|---|---|---|---|---|']
        by={(r['id'],r['arm']):r for r in p['results']}
        for q in p['sample_ids']:
            r=by[q,'harness'];cells=[]
            for arm in p['budgets']:
                v=by[q,arm];cells.append('capacity_denied' if v['status']=='native_capacity_denied' else 'correct' if v['correct'] is True else 'UNKNOWN' if v['unknown'] else 'incorrect')
            lines.append(f'| {q} | {r["domain"]} | {r["ability"]} | {r["question_image"]} | '+' | '.join(cells)+' |')
        lines+=['','## '+t('Benchmark 来源与版本','Benchmark provenance'),' ',
            '[Official repository](https://github.com/xiaowu0162/LongMemEval-V2) · [Paper](https://arxiv.org/abs/2605.12493) · [Official dataset](https://huggingface.co/datasets/xiaowu0162/longmemeval-v2)','',
            f'Code: `{p["official_code_commit"]}`; data: `{p["data_revision"]}`; freeze SHA-256: `{p["freeze_sha256"]}`.','',
            t('上游小文件 checksums 清单不包含大型截图归档；两份归档按固定版本的官方 LFS 对象 SHA 校验，5095 张历史原图与 4 张题图另逐张核对。代码采用上游 Apache-2.0 许可，见第三方声明；不据此推定来源数据的再分发权。',
              'The upstream small-file checksum manifest omits large screenshot archives; both archives were checked against pinned official LFS object hashes, then all 5,095 history images and four query images were individually verified. Upstream code is Apache-2.0; see the third-party notice. This does not establish redistribution rights for source data.'),'',
            t('端到端时间含计数和推理，离线建索引另存；不等于官方 memory-query 延迟。只报告 Qwen 用量，不推算美元费用。当前只覆盖 small、两个共享历史及一轮生成，不能外推到完整榜单或所有模型。',
              'End-to-end time includes counting and generation; offline indexing is recorded separately. It is not the official memory-query latency. Qwen usage is reported without a dollar estimate. Only small tier, two shared histories, and one run are covered; results do not establish full-leaderboard or cross-model performance.'),'',
            f'[Results JSON](../benchmarks/{ID}.json) · [Offline audit](../benchmarks/{ID}_audit.json) · [Third-party notice](../third_party/longmemeval_v2/README.md)','']
        suffix='.zh-CN' if zh else ''
        (ROOT/'docs'/f'{ID}{suffix}.md').write_text('\n'.join(lines))


if __name__=='__main__':main()
