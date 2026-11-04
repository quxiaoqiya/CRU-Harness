"""Publish privacy-safe full-run tables; also maintain a local live status file."""
from collections import Counter
import csv
import json
from pathlib import Path
import time

from scripts import benchmark_longmemeval_v2_full as f

b=f.b
REPORT=b.ROOT/'docs/longmemeval_v2_full_20261104.zh-CN.md'


def live(work,phase='inference_running'):
    work=Path(work);out=work/'formal'
    records=[json.loads(p.read_text()) for p in (out/'predictions').glob('*.json')]
    physical=[r for r in records if not r.get('shared_physical_record')]
    counts=Counter(r['arm'] for r in physical)
    status={'updated_at_unix':time.time(),'phase':phase,'question_count':451,
        'completed_physical_jobs':len(physical),'planned_physical_jobs':931,
        'completed_logical_records':len(records),'planned_logical_records':1353,
        'physical_status_counts':dict(Counter(r['status'] for r in physical)),
        'completed_by_arm':dict(counts),
        'completed_physical_generations':sum(r['usage']['physical_requests'] for r in physical),
        'completed_physical_tokens':sum(r['usage']['charged_tokens'] for r in physical)}
    b.atomic(work/'live_status.json',status)
    text=f'''# LongMemEval-V2 Small 全量评测：运行中

更新时间：{time.strftime('%Y-%m-%d %H:%M:%S %Z')}。阶段：`{phase}`。

451 题，直连 Qwen／旧 V1／新 V3 三路，共 1353 条方法结果。
无题图的 422 题使用两版相同的文本流程，明确共享计算；实际独立任务 931 个。
6 路并发：直连 3 路，Harness 3 路；评分另有 6 路。

| 任务 | 已完成 | 总数 |
|---|---:|---:|
| 直连 Qwen | {counts['direct_headtail_native']} | 451 |
| 当前 Harness V3／共同文本路径 | {counts['harness_visual_anchor']} | 451 |
| 独立旧 V1 视觉路径 | {counts['harness_visual_anchor_v1']} | 29 |
| 独立执行任务合计 | {len(physical)} | 931 |

完成任务状态：`{dict(Counter(r['status'] for r in physical))}`。
已完成任务用量：{status['completed_physical_generations']:,} 次生成，
{status['completed_physical_tokens']:,} tokens；不含仍在执行的预留及评分。

此前已用 76 题、未用 375 题将在最终报告中分开。全量推理固定后才打开评分标签。
模型为 `Qwen3.8-Flash-Next-FP8`，Small 档每题 100 条历史轨迹；来源历史复用。
当前还没有全量分数。完成评分与审计后，此文件将由正式报告替换。

所有编排和落盘都在本地 Mac；没有修改远端文件。检查点位于
`work/longmemeval_v2_full_qwen_20261104/formal/`。
'''
    REPORT.write_text(text)
    return status


def paired(rows,left,right):
    a={r['id']:r for r in rows if r['arm']==left};z={r['id']:r for r in rows if r['arm']==right}
    assert set(a)==set(z)
    return {'n':len(a),'both_correct':sum(a[k]['correct'] is True and z[k]['correct'] is True for k in a),
        'new_gain':sum(a[k]['correct'] is not True and z[k]['correct'] is True for k in a),
        'new_loss':sum(a[k]['correct'] is True and z[k]['correct'] is not True for k in a),
        'both_wrong':sum(a[k]['correct'] is not True and z[k]['correct'] is not True for k in a)}


def export(work):
    work=Path(work);frozen=f.check(work);out=work/'formal'
    audit=json.loads((out/'audit.json').read_text());offline=json.loads((work/'offline_audit.json').read_text())
    if offline['status']=='integrity_mismatch':raise ValueError('Do not publish failed integrity audit')
    assert offline['terminal_records']==1353
    rows=[]
    for r in audit['scored']:
        rows.append({'id':r['id'],'domain':r['domain'],'arm':r['arm'],'question_type':r['question_type'],
            'cohort':r['cohort'],'question_image':r['question_image'],'status':r['status'],
            'correct':r['correct'],'unknown':r['unknown'],'judge_error':r['judge_error'],
            'scoring_function':r['scoring_function'],'method_assigned_tokens':r['usage']['charged_tokens'],
            'method_assigned_generations':r['usage']['physical_requests'],'within_budget':r['usage']['within_budget'],
            'physical_compute_shared':bool(r.get('shared_physical_record')),
            'selected_evidence_omitted':len(r['result'].get('pinned_omitted',[])),
            'selected_visual_omitted':len(r['result'].get('final_visual_omitted',[])),
            'record_sha256':b.digest(b.record_path(out,r['id'],r['arm']))})
    physical=[r for r in audit['scored'] if not r.get('shared_physical_record')]
    judges=[e for r in physical for e in r['judge_ledger']]
    results={'schema_version':1,'model':b.MODEL,'haystack_tier':'small','n_questions':451,
        'logical_records':1353,'physical_records':931,'shared_text_records':422,
        'channels':frozen['channels'],'freeze_sha256':b.digest(out/'freeze.json'),
        'code_sha256':frozen['code_sha256'],'upstream_code_commit':frozen['upstream_code_commit'],
        'data_revision':frozen['data_revision'],'budgets':frozen['budgets'],
        'source_histories_reused':True,'official_leaderboard_submission':False,
        'judge_substitution_from_official_default':True,'same_compute_comparison':False,
        'all_inference_fixed_before_labels':True,'summary':audit['summary'],'rows':rows,
        'physical_tokens':sum(r['usage']['charged_tokens'] for r in physical),
        'physical_generations':sum(r['usage']['physical_requests'] for r in physical),
        'physical_judge_tokens':sum(e['charged_tokens'] for e in judges),'physical_judge_calls':len(judges),
        'paired_vs_v1':paired(rows,'harness_visual_anchor_v1','harness_visual_anchor'),
        'paired_vs_direct':paired(rows,'direct_headtail_native','harness_visual_anchor'),
        'offline_audit':{k:v for k,v in offline.items() if k not in ('receipts',)},
        'offline_audit_sha256':b.digest(work/'offline_audit.json')}
    target=b.ROOT/'benchmarks/longmemeval_v2_full_20261104.json';b.atomic(target,results)
    with target.with_suffix('.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    names={'direct_headtail_native':'直连 Qwen（首尾截取）','harness_visual_anchor_v1':'旧 V1','harness_visual_anchor':'新 V3'}
    def value(s):return f"{s['correct']}/{s['n']}（{s['accuracy_all_planned']:.2%}）" if s['n'] else '—'
    lines=['# LongMemEval-V2 Small 全量三路评测报告','',
        '本轮覆盖固定版本全部 451 道题，每题 100 条历史轨迹。此前已用 76 题、未用 375 题分开报告。',
        '其中 29 道带截图，422 道无题图。V1 与 V3 的无图回退代码及参数完全相同，422 条结果明确共享；不是两次独立随机复现。','',
        '| 路径 | 全部451 | 未用375 | 已用76 | 视觉29 | 方法计量tokens | 方法生成次数 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for arm in f.ARMS:
        s=audit['summary'][arm]
        lines.append(f"| {names[arm]} | {value(s['overall'])} | {value(s['by_cohort']['unused_question'])} | {value(s['by_cohort']['previously_consumed'])} | {value(s['visual'])} | {s['overall']['method_assigned_tokens']:,} | {s['overall']['method_assigned_generations']} |")
    lines+=['','方法计量表示该方法单独执行的用量，不能相加充当实际总调用；共享文本计算仅计费一次。',
        f"实际推理 {results['physical_generations']:,} 次、{results['physical_tokens']:,} tokens；裁判另计 {len(judges):,} 次、{results['physical_judge_tokens']:,} tokens。",'',
        '## 按能力分类','', '| 类型 | 题数 | 直连 | V1 | V3 |','|---|---:|---:|---:|---:|']
    types=audit['summary'][f.ARMS[0]]['by_question_type']
    for kind in sorted(types):
        s=[audit['summary'][a]['by_question_type'][kind] for a in f.ARMS]
        lines.append(f"| {kind} | {s[0]['n']} | {value(s[0])} | {value(s[1])} | {value(s[2])} |")
    lines+=['','`*-abs` 测的是识别并解释错误前提；仅输出 UNKNOWN 按上游规则记错，不能视为正确拒答。','',
        '## 成对得失','',f"相对旧 V1：{results['paired_vs_v1']}。",f"相对直连：{results['paired_vs_direct']}。",'',
        '## 逐题结果','', '| ID | 域 | 能力 | 已用/未用 | 直连 | V1 | V3 |','|---|---|---|---|---|---|---|']
    by={(r['id'],r['arm']):r for r in rows}
    for qid in sorted({r['id'] for r in rows}):
        group=[by[qid,a] for a in f.ARMS];sample=group[0]
        flags=['待定（裁判失败）' if r['correct'] is None else '对' if r['correct'] else '错' for r in group]
        lines.append('| '+ ' | '.join([qid,sample['domain'],sample['question_type'],sample['cohort'],*flags])+' |')
    lines+=['','## 边界与审计','',
        '- 直连上限 262,144 tokens／1次生成，采用原文首尾截取；不是完整历史直连。Harness 单次 16,384、每题累计 65,536、最多8次生成；输出均最多4096。',
        '- 模型 Qwen3.8-Flash-Next-FP8，temperature0、seed20261103、thinking关闭；同模型不同预算配置。索引CPU和下载成本不折算成模型tokens。',
        '- 数据来源 [官方代码](https://github.com/xiaowu0162/LongMemEval-V2/tree/2cc8c540bdb87fe6761629b585e727e1c4704520) 与 [固定数据版本](https://huggingface.co/datasets/xiaowu0162/longmemeval-v2/tree/f152293e235517d504809563c833d7190b8c713b)。',
        '- 复用两份历史；未用问题不代表来源或模板独立。视觉29题中28题此前已用，不能把视觉全量当作未见验证。',
        '- 沿用官方匹配／LLM评分规则，Qwen替代默认裁判。不是官方榜单提交或LAFS分数；得分不证明所有附加解释都有证据。',
        '- 失败记录、UNKNOWN和未解析裁判不从计划分母剔除；裁判失败另列为待定，未静默重试不确定推理。',
        f"- 离线审计状态：`{offline['status']}`。独立任务重放通过 {offline['totals'].get('physical_replays_passed',0)}/931；无法重放 {offline['totals'].get('physical_replays_unavailable',0)}；共享结果422。审计没有调用服务或tokenizer。",
        f"- 冻结SHA：`{results['freeze_sha256']}`。本地Mac编排落盘，未修改远端文件。",'',
        '[完整脱敏JSON](../benchmarks/longmemeval_v2_full_20261104.json) · [1353条CSV](../benchmarks/longmemeval_v2_full_20261104.csv)',
        '','私有原始输出、标签、账本和来源保存在 `work/longmemeval_v2_full_qwen_20261104/`。']
    REPORT.write_text('\n'.join(lines)+'\n')
    live_status={'phase':'complete','report':str(REPORT),'logical_records':1353,'physical_records':931,'updated_at_unix':time.time()}
    b.atomic(work/'live_status.json',live_status)
    print(json.dumps({'report':str(REPORT),'summary':audit['summary']},ensure_ascii=False,indent=2))
    return results


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--live',action='store_true');parser.add_argument('--work',type=Path,default=f.WORK)
    args=parser.parse_args();(live if args.live else export)(args.work)
