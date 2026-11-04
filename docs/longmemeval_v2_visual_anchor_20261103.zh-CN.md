# LongMemEval-V2 视觉题专项：题图状态与操作结果检索

本轮只评测带题图的 gotchas 题。12 道此前已使用的视觉题用于开发，另选 8 道新视觉题作问题留出；与此前 60 个问题 ID／规范化正文哈希零重叠。仍复用两份 100 轨迹历史，不是来源独立或任务模板独立的评测。

## 新视觉题结果

| Arm | Correct | UNKNOWN | Inference tokens | Generations |
|---|---:|---:|---:|---:|
| direct_headtail_native | 2/8 (25.0%) | 0 | 1,996,847 | 8 |
| harness_completion | 1/8 (12.5%) | 1 | 267,907 | 24 |
| visual_query_only | 0/8 (0.0%) | 1 | 24,465 | 16 |
| harness_visual_anchor | 2/8 (25.0%) | 0 | 197,643 | 26 |

新候选保留原完成版答对的 1/1 题，新增 1 题，损失 0 题；推理 tokens 少 26.23%。但仍有 6/8 题答错，当前正确率不足以支持普遍视觉优势或可靠拒答的结论。

题图哈希与开发题图重叠 0 张；该检查不排除共享历史、相同界面模板或视觉近似。


推理用量不含裁判，裁判另计。三个 16k 路径共享单次 16,384／累计 65,536／最多 8 次生成／最多 4096 输出的硬上限；实际用量分别报告。直连首尾使用更大的 262,144 单次／累计、1 次生成，并保留最多 16 张原图。

企业域四条路径均为 0/4；网页域直连和新候选均为 2/4。题图感知 JSON 解析通过不证明识别内容正确，也不能区分所有错题究竟来自召回不足还是推理错误。

## 实现与对照含义

新题在揭晓结果前固定使用第一版视觉召回候选：开发集 7/12，高于第二版 6/12。第二版修复星级控件题，但在报告模块和过滤器题回归，未作为新题候选；其代码、分数与完整审计一并保留。

先看原始题图，生成可见标签、状态值和检索提示，不提前生成最终答案。题图感知笔记作为检索提示，最终仍附原图。历史路径检索观察字段及“前一状态控件→记录动作→后一状态”成对证据。文字相同但图片不同的状态保持独立，固定相关证据时同时保留配对端点。最终使用原生推理，不施加自报置信度硬门槛。

visual_query_only 使用相同题图感知与最终策略，但没有任何历史证据，可判断收益是否仅来自更仔细读题图。它与完整路径逐题对照，不能将其收益全部归于记忆召回。当前题图状态优先用于判断当前操作前提；相似历史页面不能覆盖当前状态。

## 成对得失

| Control | Both correct | Anchor only | Control only | Neither |
|---|---:|---:|---:|---:|
| harness_completion | 1 | 1 | 0 | 6 |
| visual_query_only | 0 | 2 | 0 | 6 |

## 开发回归

### 第一版：新题选定候选

| Arm | Correct/12 | UNKNOWN | Inference tokens |
|---|---:|---:|---:|
| harness_completion | 6/12 | 0 | 350,590 |
| visual_query_only | 4/12 | 1 | 42,510 |
| harness_visual_anchor | 7/12 | 0 | 351,982 |

### 第二版：回归后未选用

| Arm | Correct/12 | UNKNOWN | Inference tokens |
|---|---:|---:|---:|
| harness_completion | 6/12 | 1 | 320,292 |
| visual_query_only | 2/12 | 0 | 44,236 |
| harness_visual_anchor | 6/12 | 0 | 431,564 |

第一版开发对照为完成版 6/12、只读题图 4/12、视觉召回 7/12，保留全部记录；后续修正加入目标状态属性变化、聚焦控件及目标页面限定词核对。


开发成绩已参与调优，不能作为独立泛化成绩。原默认控制器及以前冻结结果保持原样，新模块显式启用。评测在 Mac 编排执行，未修改远端推理服务文件；代码和脱敏结果在评测结束后另行发布。

## 逐题结果与诊断

| ID | Domain | Direct | Completion | Query only | Visual anchor |
|---|---|---|---|---|---|
| 7586cf7c | enterprise | incorrect | incorrect | incorrect | incorrect |
| 285db8ec | enterprise | incorrect | incorrect | incorrect | incorrect |
| 50d92f55 | enterprise | incorrect | incorrect | incorrect | incorrect |
| 18b91103 | enterprise | incorrect | incorrect | incorrect | incorrect |
| f2b221fd | web | correct | incorrect | incorrect | incorrect |
| 81180ae9 | web | incorrect | incorrect | incorrect | incorrect |
| 626e401e | web | correct | UNKNOWN | UNKNOWN | correct |
| b339209c | web | incorrect | correct | incorrect | correct |

harness_visual_anchor: transition cases 2/8; perception parse errors 0.

visual_query_only: transition cases 0/8; perception parse errors 0.

轨迹状态说明与实际记录的动作对齐存在歧义。本地诊断检查 594 次填充：同一控件两端均存在的 531 次中，495 次输入值只在后一状态出现，反向为 0。因此本适配器将 state i 的记录动作与 i-1→i 配对；该诊断不证明每个操作均成功。

## 审计与适用范围

本轮 56 项针对性回归通过。第一版开发 36/36、第二版开发 36/36、留出 32/32 控制流程离线重放通过，核对图片原始字节、来源字段及成对状态关系。所有本阶段输出都在本阶段评分标签访问前固定。机械完整性不等于人工语义相关性／召回标注。

模型 Qwen3.8-Flash-Next-FP8，temperature=0、seed=20261103、thinking 关闭。固定官方数据及评分提示；新题中 7 道使用 Qwen 替代默认 gpt-5.2 的 LLM 裁判，1 道使用官方确定性短语匹配。8 道新题、两份共享历史、单次运行，不足以证明普遍或显著优势；不是整体 LongMemEval-V2、官方 LAFS 或线上榜单分数。

第一版开发评分首次因缺少 question_type 报告元数据停止，使用独立元数据适配脚本恢复；仅为该字段设定 gotchas 默认值。推理结果、参考答案、官方评分提示及已完成的裁判请求均保持原样。新题元数据在冻结前补齐。

[Official code](https://github.com/xiaowu0162/LongMemEval-V2) · [Dataset](https://huggingface.co/datasets/xiaowu0162/longmemeval-v2)

Data `f152293e235517d504809563c833d7190b8c713b`; upstream `2cc8c540bdb87fe6761629b585e727e1c4704520`.

```bash
python3 -S scripts/verify_longmemeval_v2_visual_anchor_results.py
```

[Results JSON](../benchmarks/longmemeval_v2_visual_anchor_20261103.json) · [Version1 development audit](../benchmarks/longmemeval_v2_visual_anchor_20261103_preliminary_development_audit.json) · [Version 2 development audit](../benchmarks/longmemeval_v2_visual_anchor_20261103_development_audit.json) · [Holdout audit](../benchmarks/longmemeval_v2_visual_anchor_20261103_holdout_audit.json) · [Opt-in API](visual_anchor_memory.md)
