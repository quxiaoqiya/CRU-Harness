"""Export only derived visual evaluation metadata after both offline audits."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[2]
ID = 'longmemeval_v2_visual_anchor_20261103'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ledger(entries):
    keys = ['physical_request', 'prompt_sha256', 'admission_prompt_tokens',
            'requested_output_tokens', 'charged_tokens', 'status', 'usage_known', 'finish_reason']
    return [{**{k:e.get(k) for k in keys},
             'input_tokens': e['usage']['prompt_tokens'] if e.get('usage_known') else None,
             'output_tokens': e['usage']['completion_tokens'] if e.get('usage_known') else None}
            for e in entries]


def export_stage(stage):
    names={'preliminary_development':'longmemeval_v2_visual_anchor_dev_20261103','development':'longmemeval_v2_visual_anchor_v2_dev_20261103','holdout':'longmemeval_v2_visual_anchor_holdout_20261103'}
    work = ROOT/'work'/names[stage]
    out = work/'formal'
    frozen = json.loads((out/'freeze.json').read_text())
    audit = json.loads((work/'offline_audit.json').read_text())
    scored = json.loads((out/'audit.json').read_text())
    access = json.loads((out/'scoring_access.json').read_text())
    selection = json.loads((work/'selection_manifest.json').read_text())
    assert audit['status'] == 'pass' and not audit['failures']
    assert audit['replay_passed'] == frozen['expected_terminal_records']
    rows = []
    for r in scored['scored']:
        z = r['result']
        events = z.get('image_events', [])
        p = z.get('perception', {})
        query_hashes={im['image_sha256'] for event in events for im in event['images'] if im.get('query_image')}
        assert len(query_hashes)==1
        rows.append({k:r[k] for k in ['id','arm','domain','question_image','status','correct','unknown','judge_error','scoring_function','completed_at_unix']})
        rows[-1].update(model=r['model'], actual_models=sorted({e['actual_model'] for e in r['response_events']}),
            source_sha256=r['source_metadata']['canonical_source_sha256'], source_text_tokens=r['source_text_tokens_exact'],
            record_sha256=digest(out/'predictions'/f'{r["id"]}__{r["arm"]}.json'),
            answer_sha256=hashlib.sha256(r['answer'].encode()).hexdigest(),
            query_image_sha256=next(iter(query_hashes)),
            calls=r['usage']['physical_requests'], tokens=r['usage']['charged_tokens'],
            within_budget=r['usage']['within_budget'], budget=r['usage']['budget'],
            ledger=ledger(r['ledger']), judge_ledger=ledger(r['judge_ledger']),
            perception_parse_error=p.get('parse_error'), visible_state_fact_count=len(p.get('visible_state_facts',[])),
            perception_sha256=hashlib.sha256(json.dumps(p,sort_keys=True,ensure_ascii=False).encode()).hexdigest() if p else None,
            transition_pairs=sum(len(e.get('transition_receipts',[])) for e in events),
            history_image_occurrences=sum(not im.get('query_image',False) for e in events for im in e['images']),
            query_image_occurrences=sum(im.get('query_image',False) for e in events for im in e['images']),
            final_evidence_ids=z.get('final_evidence_ids',[]), pinned_evidence_ids=z.get('pinned_evidence_ids',[]),
            pinned_omitted=z.get('pinned_omitted',[]), no_confidence_gate=z.get('no_confidence_gate'),
            truncated_generations=sum(e.get('finish_reason')=='length' for e in r['ledger']))
    summary = {}
    for arm in frozen['arms']:
        group = [r for r in rows if r['arm']==arm]
        summary[arm] = {'n':len(group),'correct':sum(r['correct'] is True for r in group),
            'accuracy':sum(r['correct'] is True for r in group)/len(group),
            'unknown':sum(r['unknown'] for r in group),'tokens':sum(r['tokens'] for r in group),
            'calls':sum(r['calls'] for r in group),
            'transition_cases':sum(r['transition_pairs']>0 for r in group),
            'perception_errors':sum(r['perception_parse_error'] is True for r in group),
            'domain':{d:{'n':sum(r['domain']==d for r in group),'correct':sum(r['correct'] is True and r['domain']==d for r in group)} for d in ['enterprise','web']}}
    by={(r['id'],r['arm']):r for r in rows}
    paired={}
    for control in ['harness_completion','visual_query_only']:
        pairs=[(by[q,control]['correct'],by[q,'harness_visual_anchor']['correct']) for q in frozen['question_ids']]
        paired[control]=dict(zip(['both_correct','anchor_only','control_only','neither'],[pairs.count(x) for x in [(True,True),(False,True),(True,False),(False,False)]]))
    target=ROOT/'benchmarks'/f'{ID}_{stage}_audit.json'
    shutil.copyfile(work/'offline_audit.json', target)
    adapter = work/'scoring_metadata_adapter.json'
    return {'stage':stage,'visual_contracts_version':frozen.get('visual_contracts_version',1),'n_questions':len(frozen['question_ids']),'arms':frozen['arms'],
        'posthoc_scoring_metadata_adapter':json.loads(adapter.read_text()) if adapter.exists() else None,
        'sample_ids':frozen['question_ids'],'outcomes':len(rows),'results':rows,'summary':summary,'paired':paired,
        'freeze_sha256':digest(out/'freeze.json'),'code_sha256':frozen['code_sha256'],
        'frozen_at_unix':frozen['frozen_at_unix'],'first_scoring_access_unix':access['opened_at_unix'],
        'all_outcomes_fixed_before_scoring':all(r['completed_at_unix']<access['opened_at_unix'] for r in rows),
        'budgets':frozen['budgets'],'judge_budget':frozen['judge_budget'],'anchor_config':frozen['anchor_config'],
        'question_id_overlap_with_prior60':selection['question_id_overlap_with_prior60'],
        'question_text_hash_overlap_with_prior60':selection['normalized_question_hash_overlap_with_prior60'],
        'offline_audit_sha256':digest(target),'auditor_sha256':audit['auditor_sha256'],
        'replay_passed':audit['replay_passed'],'judge_calls':sum(len(r['judge_ledger']) for r in rows),
        'judge_tokens':sum(e['charged_tokens'] for r in rows for e in r['judge_ledger'])}


def report(p):
    dev, held = p['experiments']['development'], p['experiments']['holdout']
    for zh in [False, True]:
        def t(a,b):return a if zh else b
        lines=['# '+t('LongMemEval-V2 视觉题专项：题图状态与操作结果检索','LongMemEval-V2 visual evaluation: current-state perception and action outcomes'),'',
            t('本轮只评测带题图的 gotchas 题。12 道此前已使用的视觉题用于开发，另选 8 道新视觉题作问题留出；与此前 60 个问题 ID／规范化正文哈希零重叠。仍复用两份 100 轨迹历史，不是来源独立或任务模板独立的评测。',
              'This evaluation covers screenshot-bearing gotchas questions only. Twelve previously consumed visual questions form development; eight new visual questions exclude the prior 60 IDs/normalized texts. The same two 100-trajectory histories are reused; this is not source- or template-disjoint.'),'',
            '## '+t('新视觉题结果','New visual-question results'),'','| Arm | Correct | UNKNOWN | Inference tokens | Generations |','|---|---:|---:|---:|---:|']
        for arm,s in held['summary'].items():lines.append(f'| {arm} | {s["correct"]}/{s["n"]} ({s["accuracy"]:.1%}) | {s["unknown"]} | {s["tokens"]:,} | {s["calls"]} |')
        base=held['summary']['harness_completion'];anchor=held['summary']['harness_visual_anchor']
        pair=held['paired']['harness_completion']
        lines+=['',t(f'新候选保留原完成版答对的 {pair["both_correct"]}/{base["correct"]} 题，新增 {pair["anchor_only"]} 题，损失 {pair["control_only"]} 题；推理 tokens 少 {1-anchor["tokens"]/base["tokens"]:.2%}。但仍有 {anchor["n"]-anchor["correct"]}/{anchor["n"]} 题答错，当前正确率不足以支持普遍视觉优势或可靠拒答的结论。',
            f'The candidate retains {pair["both_correct"]}/{base["correct"]} baseline-correct questions, gains {pair["anchor_only"]}, loses {pair["control_only"]}, and uses {1-anchor["tokens"]/base["tokens"]:.2%} fewer inference tokens. It still misses {anchor["n"]-anchor["correct"]}/{anchor["n"]}; these results do not establish general visual superiority or reliable abstention.'),'',
            t(f'题图哈希与开发题图重叠 {p["holdout_query_image_hash_overlap_with_development"]} 张；该检查不排除共享历史、相同界面模板或视觉近似。',
              f'Exact query-image hash overlap with development: {p["holdout_query_image_hash_overlap_with_development"]}. This does not exclude shared histories, UI templates or perceptually similar images.'),'']
        lines+=['',t('推理用量不含裁判，裁判另计。三个 16k 路径共享单次 16,384／累计 65,536／最多 8 次生成／最多 4096 输出的硬上限；实际用量分别报告。直连首尾使用更大的 262,144 单次／累计、1 次生成，并保留最多 16 张原图。',
                    'Inference usage excludes separately metered scoring. Three 16k arms share 16,384/request, 65,536/task, eight generations and 4,096 maximum output; actual usage differs. Head/tail Direct has larger 262,144 request/task ceilings, one generation and up to 16 original images.'),'',
            t('企业域四条路径均为 0/4；网页域直连和新候选均为 2/4。题图感知 JSON 解析通过不证明识别内容正确，也不能区分所有错题究竟来自召回不足还是推理错误。',
              'All four arms score 0/4 in the enterprise domain; Direct and the candidate score 2/4 on web questions. Successful perception-JSON parsing does not establish accurate perception or separate every retrieval failure from reasoning failure.'),'',
            '## '+t('实现与对照含义','Implementation and controls'),'',
            t('新题在揭晓结果前固定使用第一版视觉召回候选：开发集 7/12，高于第二版 6/12。第二版修复星级控件题，但在报告模块和过滤器题回归，未作为新题候选；其代码、分数与完整审计一并保留。',
              'The holdout candidate is fixed to visual-recall version 1 before any new result: development 7/12 versus version 2 at 6/12. Version 2 fixes the rating-widget case but regresses on report-module and filter questions, so it is not selected. Its code, scores and full audit are retained.'),'',
            t('先看原始题图，生成可见标签、状态值和检索提示，不提前生成最终答案。题图感知笔记作为检索提示，最终仍附原图。历史路径检索观察字段及“前一状态控件→记录动作→后一状态”成对证据。文字相同但图片不同的状态保持独立，固定相关证据时同时保留配对端点。最终使用原生推理，不施加自报置信度硬门槛。',
              'An original-query-image pass extracts visible labels, state values and retrieval hints without answering. Generated notes route retrieval; final inference still receives the original image. Historical retrieval uses observed fields and before-control/action/after-state pairs. Identical text with different images retains distinct states; pinning a transition retains both endpoints. Final inference uses native reasoning without a confidence gate.'),'',
            t('visual_query_only 使用相同题图感知与最终策略，但没有任何历史证据，可判断收益是否仅来自更仔细读题图。它与完整路径逐题对照，不能将其收益全部归于记忆召回。当前题图状态优先用于判断当前操作前提；相似历史页面不能覆盖当前状态。',
              'visual_query_only shares query perception and final policy but supplies no history, testing whether improvements come only from closer image reading. Per-question comparison prevents attributing all gains to memory. Current-image values govern current preconditions; similar historical pages cannot override them.'),'',
            '## '+t('成对得失','Paired outcomes'),'','| Control | Both correct | Anchor only | Control only | Neither |','|---|---:|---:|---:|---:|']
        for control,v in held['paired'].items():lines.append('| '+control+' | '+' | '.join(str(v[k]) for k in ['both_correct','anchor_only','control_only','neither'])+' |')
        lines+=['','## '+t('开发回归','Development regression'),'','### '+t('第一版：新题选定候选','Version1: selected for holdout'),'','| Arm | Correct/12 | UNKNOWN | Inference tokens |','|---|---:|---:|---:|']
        for arm,s in p['experiments']['preliminary_development']['summary'].items():lines.append(f'| {arm} | {s["correct"]}/12 | {s["unknown"]} | {s["tokens"]:,} |')
        lines+=['','### '+t('第二版：回归后未选用','Version 2: not selected after regressions'),'','| Arm | Correct/12 | UNKNOWN | Inference tokens |','|---|---:|---:|---:|']
        for arm,s in dev['summary'].items():lines.append(f'| {arm} | {s["correct"]}/12 | {s["unknown"]} | {s["tokens"]:,} |')
        lines+=['',t('第一版开发对照为完成版 6/12、只读题图 4/12、视觉召回 7/12，保留全部记录；后续修正加入目标状态属性变化、聚焦控件及目标页面限定词核对。','The first development version scores completion 6/12, query-only 4/12, visual recall 7/12; records are preserved. The second version adds observed target-attribute effects, focused controls and destination qualifiers.'),'']
        lines+=['',t('开发成绩已参与调优，不能作为独立泛化成绩。原默认控制器及以前冻结结果保持原样，新模块显式启用。评测在 Mac 编排执行，未修改远端推理服务文件；代码和脱敏结果在评测结束后另行发布。',
                    'Development is used for tuning, not independent generalization. Original defaults and frozen records remain unchanged; this module is opt-in. Evaluation is orchestrated on the Mac without changing inference-server files; code and sanitized results are published separately after evaluation.'),'',
            '## '+t('逐题结果与诊断','Per-question outcomes and diagnostics'),'','| ID | Domain | Direct | Completion | Query only | Visual anchor |','|---|---|---|---|---|---|']
        by={(r['id'],r['arm']):r for r in held['results']}
        for q in held['sample_ids']:
            r=by[q,'harness_visual_anchor'];cells=[]
            for arm in ['direct_headtail_native','harness_completion','visual_query_only','harness_visual_anchor']:
                row=by[q,arm];cells.append('correct' if row['correct'] else 'UNKNOWN' if row['unknown'] else 'incorrect')
            lines.append(f'| {q} | {r["domain"]} | '+' | '.join(cells)+' |')
        for arm in ['harness_visual_anchor','visual_query_only']:
            s=held['summary'][arm];lines+=['',f'{arm}: transition cases {s["transition_cases"]}/{s["n"]}; perception parse errors {s["perception_errors"]}.']
        counts=p['action_alignment_diagnostic']['counts']
        lines+=['',t(f'轨迹状态说明与实际记录的动作对齐存在歧义。本地诊断检查 {counts["fill_actions_checked"]} 次填充：同一控件两端均存在的 {counts["same_bid_both_states"]} 次中，{counts["value_appears_after_not_before"]} 次输入值只在后一状态出现，反向为 {counts["value_present_before_not_after"]}。因此本适配器将 state i 的记录动作与 i-1→i 配对；该诊断不证明每个操作均成功。',
            f'Data documentation and recorded action alignment differ. Of {counts["fill_actions_checked"]} fill actions checked, {counts["same_bid_both_states"]} retain the control on both sides; {counts["value_appears_after_not_before"]} show the literal value only after, and {counts["value_present_before_not_after"]} show the reverse. The adapter pairs the action at state i with i-1→i. This diagnostic does not prove every action succeeds.'),'',
            '## '+t('审计与适用范围','Audit and scope'),'',
            t(f'本轮 56 项针对性回归通过。第一版开发 36/36、第二版开发 {dev["replay_passed"]}/{dev["outcomes"]}、留出 {held["replay_passed"]}/{held["outcomes"]} 控制流程离线重放通过，核对图片原始字节、来源字段及成对状态关系。所有本阶段输出都在本阶段评分标签访问前固定。机械完整性不等于人工语义相关性／召回标注。',
              f'All 56 targeted regression checks pass. Offline replays pass {dev["replay_passed"]}/{dev["outcomes"]} second development, 36/36 preliminary development and {held["replay_passed"]}/{held["outcomes"]} holdout records, checking original image bytes, source fields and paired states. Each stage fixes all outcomes before its scoring-label access. Mechanical integrity is not human semantic relevance/recall annotation.'),'',
            t('模型 Qwen3.8-Flash-Next-FP8，temperature=0、seed=20261103、thinking 关闭。固定官方数据及评分提示；本轮均为 Qwen 替代默认 gpt-5.2 的 LLM 裁判。8 道新题、两份共享历史、单次运行，不足以证明普遍或显著优势；不是整体 LongMemEval-V2、官方 LAFS 或线上榜单分数。',
              'Qwen3.8-Flash-Next-FP8, temperature=0, seed=20261103, thinking disabled. Official data/scoring prompts are pinned; these questions use Qwen instead of default gpt-5.2 as LLM judge. Eight new questions, two reused histories and one run do not establish general/significant superiority. This is not an overall LongMemEval-V2, official LAFS or leaderboard score.'),'',
            t('第一版开发评分首次因缺少 question_type 报告元数据停止，使用独立元数据适配脚本恢复；仅为该字段设定 gotchas 默认值。推理结果、参考答案、官方评分提示及已完成的裁判请求均保持原样。新题元数据在冻结前补齐。',
              'Preliminary development scoring stopped on missing question_type report metadata. A separate adapter only supplies a gotchas default for that field, preserving inference records, reference answers, official judge prompts and completed judge calls. Holdout metadata was repaired before its freeze.'),'',
            '[Official code](https://github.com/xiaowu0162/LongMemEval-V2) · [Dataset](https://huggingface.co/datasets/xiaowu0162/longmemeval-v2)','',
            f'Data `{p["data_revision"]}`; upstream `{p["upstream_code_commit"]}`.','',
            '```bash','python3 -S scripts/verify_longmemeval_v2_visual_anchor_results.py','```','',
            f'[Results JSON](../benchmarks/{ID}.json) · [Version1 development audit](../benchmarks/{ID}_preliminary_development_audit.json) · [Version 2 development audit](../benchmarks/{ID}_development_audit.json) · [Holdout audit](../benchmarks/{ID}_holdout_audit.json) · [Opt-in API](visual_anchor_memory.md)','']
        (ROOT/'docs'/f'{ID}{".zh-CN" if zh else ""}.md').write_text('\n'.join(lines))


def main():
    experiments={stage:export_stage(stage) for stage in ['preliminary_development','development','holdout']}
    prior_images={r['query_image_sha256'] for r in experiments['preliminary_development']['results']}
    held_images={r['query_image_sha256'] for r in experiments['holdout']['results']}
    p={'schema_version':1,'experiment_id':ID,'benchmark':'LongMemEval-V2 visual gotchas subset',
       'model':'Qwen3.8-Flash-Next-FP8','data_revision':'f152293e235517d504809563c833d7190b8c713b',
       'upstream_code_commit':'2cc8c540bdb87fe6761629b585e727e1c4704520',
       'experiments':experiments,'source_histories_reused':True,'n_shared_histories':2,
       'selected_holdout_candidate_version':1,'holdout_query_image_hash_overlap_with_development':len(prior_images&held_images),
       'n_trajectories':200,'n_history_images':5095,'tests_passed':56,
       'action_alignment_diagnostic':json.loads((ROOT/'work/longmemeval_v2_visual_anchor_dev_20261103/action_alignment_audit.json').read_text()),
       'official_leaderboard_submission':False,'official_LAFS_score':None,
       'independent_semantic_validation':False,'remote_inference_files_modified':False,'github_push_during_benchmark':False}
    (ROOT/'benchmarks'/f'{ID}.json').write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n')
    report(p)
    print(json.dumps({stage:e['summary'] for stage,e in experiments.items()},indent=2))


if __name__=='__main__':main()
