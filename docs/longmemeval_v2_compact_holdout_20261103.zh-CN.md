# LongMemEval-V2：原生推理预算与可逆证据展示

新 20 题：直连首尾 4/20，原 Harness 8/20，原生完成预算版 11/20，可逆展示版 11/20。

问题 ID、规范化正文哈希与此前 40 题零重叠。五种能力各 4 题、两个领域各 10 题，4 题带截图。两份 100 轨迹历史仍复用，共 200 轨迹、5095 历史原图；不是来源独立留出。所有 100 条结果固定后才启用本轮评分标签。

| 路径 | 答对／20 | UNKNOWN | 覆盖率 | 已作答准确率 | 输入＋输出 tokens |
|---|---:|---:|---:|---:|---:|
| direct_full | capacity denied | 0 | 0% | — | 0 |
| direct_headtail_native | 4/20 (20%) | 8 | 60% | 33.3% | 5,002,380 |
| harness | 8/20 (40%) | 1 | 95% | 42.1% | 650,438 |
| harness_completion | 11/20 (55%) | 2 | 90% | 61.1% | 689,829 |
| harness_compact | 11/20 (55%) | 2 | 90% | 61.1% | 685,870 |

完整直连的容量拒绝不属于语义答错。本轮直连最多 4096 输出，原生请求／累计 262,144 tokens，1 次生成。各 Harness 共享单次 16,384、累计 65,536、8 次生成、单次最多 4096 的硬上限。旧控制器实际请求 1024；完成版在不删证据且可准入时分配 4096／2048／1024。更高输出分配本身就是被测机制，实际用量单列。

两份完整 canonical JSON 历史文字分别为 31,476,162 和 34,235,595 Qwen tokens，不包含图像像素。Harness 检索的是外部完整来源，模型每次仅接收预算内证据；这些来源长度不表示一次注意力窗口，也不等于语义准确扩展倍数。各路径共同使用按当前请求实际输出额度生成的预算提示。

模型为 Qwen3.8-Flash-Next-FP8，temperature=0、生成 seed=20261103，enable_thinking=false、preserve_thinking=false。

用量表统计各路径的推理输入＋输出，不包含评分裁判。评分另有独立账本：26 次生成、38,827 tokens，未计入被测方法任务预算。tokenizer 预检与离线重放不作为生成次数。

## 结论与尚未解决的问题

两种候选均保留原版答对的 8/8 与 8/8 题，各新增 3 和 3 道正确题。本轮完成适配组合的准确率比原版提高 15 个百分点；这一小样本观察不能唯一归因于某个单独机制。

完成版与展示版同为 11/20，但有 4 题正确性不同，未做逐题取优拼接。当前证据支持展示层节省空间，不支持宣称展示变换另增准确率。两个新版的视觉子集均只有 1/4；本轮没有发生通过新增原图＋短引成功救回 UNKNOWN，拒答语义校准仍未完成。

## 改动与约束

保留原生推理顺序，不强迫先写答案，不使用自报置信度门控。原检索／review 复用旧实现。资源提示按当前请求真实输出额度计数及生成，避免用较大硬上限误导中间请求。已有非 UNKNOWN boxed 答案不再被二次裁判覆盖。缺失格式时最多补全一次；UNKNOWN 仅在新增状态原图可准入、且新回答附带的短引确实存在于相应观察字段时允许恢复。失败保留原候选。

可逆展示只用于最终证据：展开 JSON 换行、制表和引号转义；同一来源位置的重叠片段合并为区间并集，保留原窗口别名。不同位置的相同文本不去重，不改变观察的记录身份与数量。展示映射可还原原始来源字符，原图不改；不是生成式摘要或模型注意力窗口扩容。

引用校验只能证明短引存在于观察字段，不能证明该短引足以蕴含答案。UNKNOWN 数量和已作答准确率均公开；当前实验没有独立人工的视觉相关性标注，不能把图片完整性核对称为视觉语义召回率提升。

实际发生展示变换的完整提示，匹配计数合计 293,256 → 275,244 tokens，节省 18,012（6.14%）。其中 9 次请求的原表示在相同输出保留量下无法准入，展示后可以准入。删除的重复来源字符为 5,280；若为零，不能将收益归因于重叠合并。

该比例仅针对发生展示变换的匹配完整提示，不是端到端总用量下降率。追检次数、证据路径和实际输出可能变化，端到端输入＋输出用量以上表为准。

完整提示使用准入计数，各含 16 tokens 安全余量；配对差值不受共同余量影响。来源文字预检是单独的 tokenizer 原始计数。

默认轨迹控制器保持原样；新增完成／展示控制器为显式实验入口，不根据这 20 题自动选路。每种方法每题只运行一次，没有逐题取优、投票挑选或测试后改写答案。

| Candidate vs original | Both correct | Candidate only | Original only | Neither |
|---|---:|---:|---:|---:|
| harness_completion | 8 | 3 | 0 | 9 |
| harness_compact | 8 | 3 | 0 | 9 |

| Candidate | Native valid boxes overridden | Recovery calls | Supplemental image cases | Final output reservations |
|---|---:|---:|---:|---|
| harness_completion | 0 | 4 | 2 | {"1024": 10, "2048": 2, "4096": 8} |
| harness_compact | 0 | 2 | 1 | {"1024": 10, "2048": 3, "4096": 7} |

| Arm | Length-truncated generations | Unboxed final answers | Recovery outcomes |
|---|---:|---:|---|
| harness | 6 | 6 | {} |
| harness_completion | 6 | 3 | {"budget_unavailable_kept_native": 1, "format_recovered": 2, "invalid_recovery_kept_native": 2, "native_answer_kept": 13, "refusal_kept_no_new_observation": 2} |
| harness_compact | 4 | 2 | {"budget_unavailable_kept_native": 1, "format_recovered": 1, "invalid_recovery_kept_native": 1, "native_answer_kept": 15, "refusal_kept_no_new_observation": 2} |

## 开发记录与正确题保留

| Candidate | Correct/20 | Old correct retained/8 | Old errors recovered/12 |
|---|---:|---:|---:|
| harness_completion | 9 | 7 | 2 |
| harness_compact | 8 | 7 | 1 |
| failed harness_format | 7 | 5 | — |
| failed harness_assist | 8 | 4 | — |

开发集复用首轮 20 题，原版 8 对、12 错；这些数字只用于开发，不能当作留出成绩。强迫先答方案的失败记录保留。另一个开发样本仅对来源文字计数，10383→9357 tokens、节省 9.88%，不是完整多模态提示的压缩比。

## 分项和逐题结果

| Subset | Direct | Original | Completion | Compact |
|---|---:|---:|---:|---:|
| Deterministic | 3/12 | 6/12 | 8/12 | 8/12 |
| Qwen judged | 1/8 | 2/8 | 3/8 | 3/8 |
| Query image | 1/4 | 0/4 | 1/4 | 1/4 |
| ability: dynamic | 0/4 | 1/4 | 2/4 | 2/4 |
| ability: gotchas | 1/4 | 0/4 | 1/4 | 1/4 |
| ability: premise_awareness | 0/4 | 2/4 | 2/4 | 2/4 |
| ability: procedure | 0/4 | 3/4 | 3/4 | 4/4 |
| ability: static | 3/4 | 2/4 | 3/4 | 2/4 |
| domain: enterprise | 2/10 | 4/10 | 6/10 | 5/10 |
| domain: web | 2/10 | 4/10 | 5/10 | 6/10 |

| ID | Domain | Ability | Image | Direct | Original | Completion | Compact |
|---|---|---|---|---|---|---|---|
| d0d61088 | enterprise | dynamic | False | UNKNOWN | incorrect | incorrect | incorrect |
| 5edd2533 | enterprise | dynamic | False | incorrect | correct | correct | correct |
| 8e21c6e5 | enterprise | gotchas | True | correct | incorrect | correct | correct |
| 2f6a21cd | enterprise | gotchas | True | incorrect | incorrect | incorrect | incorrect |
| 5849c18b | enterprise | premise_awareness | False | UNKNOWN | correct | correct | correct |
| 571b841c | enterprise | premise_awareness | False | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN |
| 025db8ef | enterprise | procedure | False | UNKNOWN | correct | correct | correct |
| 25b00876 | enterprise | procedure | False | UNKNOWN | correct | correct | correct |
| a444cec8 | enterprise | static | False | correct | incorrect | incorrect | UNKNOWN |
| 2975030d | enterprise | static | False | UNKNOWN | incorrect | correct | incorrect |
| 5a85ca0f | web | dynamic | False | incorrect | incorrect | correct | incorrect |
| 11dac74b | web | dynamic | False | UNKNOWN | incorrect | UNKNOWN | correct |
| 1e77eb4b | web | gotchas | True | incorrect | incorrect | incorrect | incorrect |
| 1d864942 | web | gotchas | True | incorrect | incorrect | incorrect | incorrect |
| 1a89c3ac | web | premise_awareness | False | incorrect | incorrect | incorrect | incorrect |
| 922a0c56 | web | premise_awareness | False | UNKNOWN | correct | correct | correct |
| b4dcf650 | web | procedure | False | incorrect | correct | correct | correct |
| 99fa7c96 | web | procedure | False | incorrect | incorrect | incorrect | correct |
| 7c8efee5 | web | static | False | correct | correct | correct | correct |
| 26a038ce | web | static | False | correct | correct | correct | correct |

## 审计和适用范围

258 项检查通过。离线审计 100/100 账本、100/100 评分重算、100/100 控制流程重放，状态 pass。核对原图、原位置区间并集、可逆还原及真实准入／输出额度，不等于独立人工语义验证。

单次 20 题、两份共享历史不证明普遍显著优势。固定 seed 和温度 0 仍可能产生服务端输出变化。各 Harness 硬上限相同、实际输入与输出分配不同；直连预算更大。原模型、原控制器和此前结果未覆盖，未改原环境依赖或远端文件。官方确定性评分和裁判提示固定，LLM 裁判使用 Qwen 代替默认 gpt-5.2；不是官方 LAFS 或线上榜单分数。

[Official code](https://github.com/xiaowu0162/LongMemEval-V2) · [Paper](https://arxiv.org/abs/2605.12493) · [Dataset](https://huggingface.co/datasets/xiaowu0162/longmemeval-v2)

Data `f152293e235517d504809563c833d7190b8c713b`; code `2cc8c540bdb87fe6761629b585e727e1c4704520`; freeze `cef31f1ac7da7b732f229d44ca22211188dda9703d9360cb0d34c2b543095ac7`.

```bash
python3 -S scripts/verify_longmemeval_v2_compact_results.py
```

[Results JSON](../benchmarks/longmemeval_v2_compact_holdout_20261103.json) · [Offline audit](../benchmarks/longmemeval_v2_compact_holdout_20261103_audit.json) · [Explicit opt-in API](trajectory_completion_experiments.md)
