#!/usr/bin/env python3
"""Score a completed frozen comparison; this is the first label-reading phase."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import statistics
import time

from scripts.benchmark_context_holdout import atomic_json, check_freeze, digest_file, read_inputs
from scripts.prepare_context_holdout import score_prediction


def bootstrap_difference(left, right, *, seed=20261101, repetitions=5000):
    differences = [a - b for a, b in zip(left, right)]
    rng = random.Random(seed)
    distribution = sorted(statistics.mean(rng.choices(differences, k=len(differences))) for _ in range(repetitions))
    return {'mean_difference': statistics.mean(differences),
            'paired_bootstrap_95_percentile_ci': [distribution[int(repetitions * .025)], distribution[int(repetitions * .975)]],
            'n_pairs': len(left), 'repetitions': repetitions}


def mechanism_summary(records):
    """Report retained state separately from semantic correctness and coverage."""
    summaries = []
    for record in records:
        result = record['result']
        trace = result.get('trace', [])
        aggregate = result.get('aggregate', {})
        state = result.get('fact_state', {})
        summaries.append({
            'id': record['id'], 'mode': result.get('mode'),
            'stop_reason': result.get('stop_reason'),
            'evidence_complete': result.get('evidence_complete'),
            'answer_is_estimate': result.get('answer_is_estimate'),
            'retained_fact_count': len(result.get('record_facts', result.get('retained_facts', []))),
            'final_fact_loss_count': result.get('final_fact_loss_count'),
            'fact_digest': state.get('fact_digest'),
            'known_count': aggregate.get('known_count'),
            'unknown_count': aggregate.get('unknown_count'),
            'scope_records': result.get('record_index', {}).get('scope_records'),
            'parser_complete': result.get('record_index', {}).get('parser_complete'),
            'semantic_coverage': result.get('semantic_coverage'),
            'operation': aggregate.get('operation'),
            'group_by': aggregate.get('group_by'),
            'reduction': aggregate.get('reduction'),
            'coverage_complete': aggregate.get('coverage_complete'),
            'information_gains': [t.get('information_gain', 0) for t in trace
                                  if t.get('action') in {'classify_records', 'fact_review'}],
            'max_stagnant_rounds': max((t.get('stagnant_rounds', 0) for t in trace), default=0),
            'deterministic_final': any(t.get('action') == 'deterministic_finalize' for t in trace),
            'final_answer_unavailable': any(t.get('action') == 'final_answer_unavailable' for t in trace),
        })
    return summaries


def analyze(args):
    frozen = check_freeze(args)
    output = Path(args.output)
    inputs = read_inputs(args.inputs)
    records = {}
    for sample in inputs:
        for method in frozen['methods']:
            path = output / 'predictions' / f'{sample["id"]}__{method}.json'
            if not path.exists():
                raise ValueError('Inference incomplete; labels remain unopened: ' + path.name)
            record = json.loads(path.read_text())
            if record['freeze_sha256'] != digest_file(output / 'freeze.json'):
                raise ValueError('Prediction freeze mismatch')
            if (record.get('id') != sample['id'] or record.get('method') != method
                    or record.get('context_sha256') != hashlib.sha256(sample['context'].encode()).hexdigest()):
                raise ValueError('Prediction identity or source mismatch')
            records[sample['id'], method] = record
    data_manifest = json.loads(Path(frozen['data_manifest']).read_text())
    if digest_file(args.labels) != frozen['labels_sha256']:
        raise ValueError('Labels differ from the pre-inference sealed hash')
    scorer = Path(args.inputs).parent / 'source/eval_helpers_original.py'
    if digest_file(scorer) != frozen['official_scorer_sha256']:
        raise ValueError('Official scorer source differs from the sealed hash')
    labels = json.loads(Path(args.labels).read_text())
    if set(labels) != set(frozen['sample_ids']):
        raise ValueError('Label identities do not match frozen inputs')
    access_path = output / 'scoring_access.json'
    first_access = json.loads(access_path.read_text()).get('labels_first_scored_at_unix') if access_path.exists() else time.time()
    atomic_json(access_path, {'labels_first_scored_at_unix': first_access, 'labels_last_scored_at_unix': time.time(),
                'all_predictions_present_before_scoring': True,
                'labels_sha256': digest_file(args.labels), 'scorer_sha256': digest_file(scorer)})
    rows = []
    for sample in inputs:
        for method in frozen['methods']:
            record = records[sample['id'], method]
            score = score_prediction(record['prediction'], labels[sample['id']])
            rows.append({'id': sample['id'], 'method': method,
                         'context_len': record['context_len'], 'task_type': record['task_type'],
                         'prediction': record['prediction'], 'gold': labels[sample['id']]['answer'],
                         **score, 'responded': bool(record['prediction'].strip()),
                         'error': record['error'], 'usage': record['usage'],
                         'seconds': record['seconds'], 'baseline_identity': record['result'].get('baseline_identity')})
    summaries = {}
    for method in frozen['methods']:
        selected = [r for r in rows if r['method'] == method]
        summaries[method] = {'n': len(selected),
            'mean_official_score': statistics.mean(r['official_score'] for r in selected),
            'full_credit': sum(r['full_credit'] for r in selected),
            'exact_match': sum(r['exact_match'] for r in selected),
            'responded': sum(r['responded'] for r in selected),
            'errors': sum(bool(r['error']) for r in selected),
            'mean_calls': statistics.mean(r['usage']['physical_requests'] for r in selected),
            'mean_charged_tokens': statistics.mean(r['usage']['charged_tokens'] for r in selected),
            'total_prompt_tokens': sum(r['usage']['prompt_tokens'] for r in selected),
            'total_completion_tokens': sum(r['usage']['completion_tokens'] for r in selected),
            'mean_seconds': statistics.mean(r['seconds'] for r in selected),
            'all_within_budget': all(r['usage']['within_budget'] for r in selected),
            'all_usage_known': all(r['usage']['all_usage_known'] for r in selected)}
    comparisons = {}
    focus = 'adaptive_v3' if 'adaptive_v3' in frozen['methods'] else 'adaptive'
    adaptive = [r['official_score'] for r in rows if r['method'] == focus]
    for method in frozen['methods']:
        if method != focus:
            comparisons[focus + '_minus_' + method] = bootstrap_difference(adaptive, [r['official_score'] for r in rows if r['method'] == method])
    by_length = {}
    for length in sorted({r['context_len'] for r in rows}):
        by_length[str(length)] = {}
        for method in frozen['methods']:
            group = [r for r in rows if r['context_len'] == length and r['method'] == method]
            by_length[str(length)][method] = {'n': len(group), 'score': statistics.mean(r['official_score'] for r in group), 'full_credit': sum(r['full_credit'] for r in group)}
    identities = {method: next((r['baseline_identity'] for r in rows if r['method'] == method and r['baseline_identity']), None) for method in frozen['methods']}
    mechanisms = mechanism_summary([records[s['id'], focus] for s in inputs])
    structured_mechanisms = {
        method: mechanism_summary([records[s['id'], method] for s in inputs])
        for method in frozen['methods'] if method in {'adaptive_v3', 'adaptive_v31'}
    }
    report = {'experiment': 'Frozen OOLONG context holdout comparison', 'freeze': frozen,
              'data_manifest': data_manifest, 'summary': summaries, 'by_nominal_length': by_length,
              'paired_comparisons': comparisons, 'focus_method': focus, 'baseline_identities': identities,
              'context_measurements': json.loads((output / 'context_measurements.json').read_text()),
              'focus_mechanisms': mechanisms,
              'structured_method_mechanisms': structured_mechanisms,
              'samples': rows}
    atomic_json(output / 'audit.json', report)
    with (output / 'samples.csv').open('w', newline='') as f:
        columns = ['id', 'method', 'context_len', 'prediction', 'gold', 'official_score', 'full_credit', 'exact_match', 'responded', 'calls', 'charged_tokens', 'seconds', 'error']
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (r['usage']['physical_requests'] if k == 'calls' else r['usage']['charged_tokens'] if k == 'charged_tokens' else r.get(k)) for k in columns})
    names = {'adaptive_v3': '结构化事实 Harness v3', 'adaptive_v31': '冻结 Harness v3.1', 'adaptive': '旧 Adaptive Harness v2', 'static': 'Static（关闭缺口追检）', 'rlm': 'RLM 官方引擎＋环境适配', 'memgpt': 'MemGPT 原控制器＋I/O适配'}
    revision = (identities.get('adaptive_v3') or {}).get('revision')
    if revision:
        names['adaptive_v3'] = '结构化事实 Harness v' + str(revision)
    md = ['# 证据缺口驱动 Harness：独立 OOLONG 本地对比审计', '',
          '本轮是冻结版本后的首次测试；输入与答案分开保存，所有方法运行完成后才评分。原 LongBench 503 题均已使用，本轮没有将它们重新命名为盲测。', '',
          '## 方法与共同资源上限', '',
          f"模型：{frozen['model']}。temperature=0，thinking关闭；单次输入加输出 {frozen['budget']['context_tokens']:,} tokens，每题每方法累计 {frozen['budget']['total_tokens']:,} tokens，最多 {frozen['budget']['max_calls']} 次物理生成请求，单次输出最多 {frozen['budget']['max_output_tokens']:,} tokens。所有递归、摘要及恢复请求都共用账本。相同的是预算上限，各方法实际消耗不同。", '',
          '单次16k是本轮主动设置的受控预算，并非Qwen原生容量。名义数据长度使用OOLONG tokenizer，实际Qwen长度见context_measurements.json。', '',
          'RLM使用固定官方引擎，适配共同预算客户端与受限本地REPL；MemGPT使用固定官方Agent.step控制器，工具协议、BM25归档和本地持久化的具体适配见audit.json的baseline_identities。均不是原论文默认栈的成绩复现。', '',
          '## 总体结果', '',
          '| 方法 | 官方均分 | 满分题 | 严格匹配 | 有响应 | 错误记录 | 平均调用 | 平均计费tokens | 平均秒 |',
          '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for m, r in summaries.items():
        md.append(f"| {names[m]} | {100*r['mean_official_score']:.2f}% | {r['full_credit']}/{r['n']} | {r['exact_match']}/{r['n']} | {r['responded']}/{r['n']} | {r['errors']} | {r['mean_calls']:.2f} | {r['mean_charged_tokens']:,.0f} | {r['mean_seconds']:.2f} |")
    md += ['', 'OOLONG数字题采用0.75**abs(error)部分分，因此官方均分不能称为准确率。“有响应”仅表示输出非空，不代表答案有效；所有失败题保留在分母。传输失败按整个预留预算保守计费，没有隐式重试。', '', '## 长度分组', '', '| OOLONG名义tokens | 方法 | 官方均分 | 满分题 |', '|---:|---|---:|---:|']
    for length, methods in by_length.items():
        for method, r in methods.items():
            md.append(f"| {int(length):,} | {names[method]} | {100*r['score']:.2f}% | {r['full_credit']}/{r['n']} |")
    md += ['', '## 自适应机制审计', '']
    adaptive_records = [records[s['id'], focus] for s in inputs]
    review_rounds = [sum(t.get('action') == 'review' for t in r['result'].get('trace', [])) for r in adaptive_records]
    if focus == 'adaptive_v3':
        md += ['| ID | 分支 | 已知/未知记录 | 保留事实 | 最终丢失事实 | 完整证据 | 估计答案 | 停止原因 |',
               '|---|---|---:|---:|---:|---|---|---|']
        for r in mechanisms:
            md.append(f"| {r['id']} | {r['mode']} | {r['known_count']}/{r['unknown_count']} | {r['retained_fact_count']} | {r['final_fact_loss_count']} | {r['evidence_complete']} | {r['answer_is_estimate']} | {r['stop_reason']} |")
        md += ['', '确定性聚合避免模型重复计算及最终裁剪事实，但不保证模型给出的语义分类正确。complete只表示已累积事实在所解析范围内足以确定结果；它不是答案正确性的独立验证。未覆盖的记录与估计值保留在JSON中。', '']
    else:
        md += [f"发生至少两轮审阅的样本：{sum(n >= 2 for n in review_rounds)}/{len(inputs)}；总审阅轮次：{sum(review_rounds)}。逐轮缺口、分配预算、原文跨度与停止原因保存在predictions下。", '']
    md += [
           '来源跨度有效只证明引用来自提供过的窗口，不构成语义蕴含证明。聚合问题的证据覆盖不足必须保留为未解决状态；最终猜测仍照常评分。', '', '## 配对差值与不确定性', '', f'| {names[focus]}减去 | 官方均分差（百分点） | 配对bootstrap 95%区间 |', '|---|---:|---|']
    for key, r in comparisons.items():
        lo, hi = r['paired_bootstrap_95_percentile_ci']
        md.append(f"| {names[key.removeprefix(focus + '_minus_')]} | {100*r['mean_difference']:+.2f} | [{100*lo:+.2f}, {100*hi:+.2f}] |")
    md += ['', f'每个输入来自不同的源context；区间按题配对重采样。{len(inputs)}题小样本只能作为先导证据，不支持普遍领先结论。', '', '## 逐题结果', '', '| ID | 名义tokens | 方法 | 输出 | Gold | 官方分 | 调用 | tokens |', '|---|---:|---|---|---|---:|---:|---:|']
    clean = lambda x: str(x).replace('|', '\\|').replace('\n', ' ')[:300]
    for r in rows:
        md.append(f"| {r['id']} | {r['context_len']:,} | {names[r['method']]} | {clean(r['prediction'])} | {clean(r['gold'])} | {r['official_score']:.6f} | {r['usage']['physical_requests']} | {r['usage']['charged_tokens']:,} |")
    md += ['', '## 结论边界与复现', '',
           '本轮数据未参与本项目先前调优，但公开数据可能进入模型训练，无法据此排除预训练污染。正式结果不用于本轮代码返修后重跑；任何后续调参应换新的封存样本。', '',
           '这一比较检验当前实现及指定资源上限，不是RLM或MemGPT原论文分数复现，也不是官方排行榜提交。MemGPT执行的是原控制器，但检索存储和工具传输有明确适配，不能据此宣称超过其所有原版配置。', '',
           'freeze.json记录源码、vendor、输入、环境和数据manifest哈希；journals记录请求前预留和请求后usage；scoring_access.json记录统一完成后评分。审计JSON含完整身份与错误；CSV便于复核逐题结果。', '']
    (output / 'audit.md').write_text('\n'.join(md))
    print(json.dumps(summaries, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', required=True)
    parser.add_argument('--labels', required=True)
    parser.add_argument('--output', required=True)
    analyze(parser.parse_args())
