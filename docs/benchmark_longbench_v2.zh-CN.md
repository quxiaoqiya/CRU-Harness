# LongBench v2：历史评测与容量审计

[English](benchmark_longbench_v2.md) | **简体中文**

CRU-Harness 历史证据窗口流程在完整 **503 题中答对 313 题（62.23%）**，直连 Qwen 为 **254/503（50.50%）**，即**多答对 59 题，端到端准确率提高 11.73 个百分点**。在最后一轮针对历史 82 题长上下文筛选集的专门重跑中，Harness 为 **50/82（60.98%）**，缓存直连基线为 **1/82（1.22%）**。

最直接的容量证据来自这轮 82 题：直连 Qwen 有 **79 个提示明确因上下文超限被拒绝**，Harness 在其中**答对 49 题（49/79，62.03%）**。这证明了系统能为直连输入上限阻断的任务恢复可用的解题路径。提升同时来自容量失败恢复、证据检索和多次推理，并未单独证明同等计算量下原模型推理能力提高。

这些是使用真实 Qwen 推理、由保留实验记录核验的**本地评测结果，尚未提交 LongBench 官方榜单**。它们评测的是早期证据窗口 Harness 流程，并非当前 V3.2 结构化事实与语义聚合路径的全量 LongBench 新重跑。V3.2 的 OOLONG 成绩独立记录于 [V3.2 评测报告](benchmark_v32.md)。

## Benchmark 来源与公开记录

LongBench v2 是 Bai 等人提出的外部 Benchmark，包含六个领域的 503 道选择题。其官方长度分层使用**词数**，并非 Qwen token 数。来源：[论文](https://arxiv.org/abs/2412.15204)、[官方项目与榜单](https://longbench2.github.io/)、[官方代码](https://github.com/THUDM/LongBench)、[官方数据集](https://huggingface.co/datasets/zai-org/LongBench-v2)。Hugging Face 的公开 split 名为 `train`；历史完整评测使用该公开 Benchmark split 的全部 503 条。

[公开结果 JSON](../benchmarks/longbench_v2_historical.json) 包含逐题预测、标准选项、来源上下文哈希、汇总结果及保留本地实验文件的哈希。只发布派生成绩，不分发原始上下文、完整题文、模型轨迹、凭据和私人机器路径。历史记录未保存数据 revision 与运行前代码 commit；当前文件哈希不能追溯建立运行前冻结，因此不补造固定版本信息。

两轮结果保持独立：

- **历史完整运行**：`longbench_v2_full` 的全部 503 题。
- **最后一轮长上下文专门重跑**：`longbench_v2_over260k_uniform_v2` 的既有 82 题；最终审计没有针对单题的结果覆盖。其 Direct 记录复用早期缓存。

不把新 82 题成绩替换到旧 503 题中，拼接成同一版本全量成绩。82 题已被检查并反复用于优化，属于回归集，不是未参与调优的 holdout。

## 总体成绩

“全部准确率”为正确数除以全部尝试题数，缺失答案和请求被拒绝均保留在分母；“有答案准确率”只以解析出选项的题目为分母。

| 运行 | 方法 | 解析出答案 | 正确 | 全部准确率 | 有答案准确率 |
|---|---|---:|---:|---:|---:|
| 完整 503 题 | Direct Qwen | 401/503 | 254 | **50.50%** | 63.34% |
| 完整 503 题 | 历史 Harness | 502/503 | 313 | **62.23%** | 62.35% |
| 长上下文 82 题重跑 | 缓存 Direct Qwen | 3/82 | 1 | **1.22%** | 33.33% |
| 长上下文 82 题重跑 | 改进检索与验证的历史 Harness | 81/82 | 50 | **60.98%** | 61.73% |

完整运行的 102 个 Direct 失败、82 题重跑的 79 个 Direct 失败均有明确的上下文超限错误。它们是直连完整提示路径的交付失败，不能描述为观察到模型答错 102 或 79 题。Harness 在完整运行被拒绝的题目中答对 59/102，在重跑子集中答对 49/79。

完整运行有 **394 题 `direct_pass` 复用缓存 Direct 输出**，其余为 **98 题 `raw_source`** 与 **11 题 `evidence_fallback`**；82 题重跑为 **81 题 `raw_source`** 和 **1 题缓存 `direct_pass`**。因此许多 Direct/Harness 单元共享同一次生成，这是配对执行路径比较，不能视为完全独立的重复推理。

82 题重跑相对缓存 Direct **多覆盖 78 题、多答对 49 题**。选项验证覆盖 **81 题、242 个视图**，发生 **1 次解析重试**，保留 **1 个解析或请求错误**，没有记录选项验证覆盖原答案的情况。单个验证视图失败与缺失最终答案是不同概念；Harness 仍有 1 题未解析出选项。

## 完整 503 题：按领域

| 领域 | 题数 | Direct 正确 / 全部准确率 | Harness 正确 / 全部准确率 |
|---|---:|---:|---:|
| Code Repository Understanding | 50 | 13/50 · 26.00% | 32/50 · 64.00% |
| Long In-context Learning | 81 | 39/81 · 48.15% | 55/81 · 67.90% |
| Long Structured Data Understanding | 33 | 12/33 · 36.36% | 19/33 · 57.58% |
| Long-dialogue History Understanding | 39 | 30/39 · 76.92% | 30/39 · 76.92% |
| Multi-Document QA | 125 | 63/125 · 50.40% | 69/125 · 55.20% |
| Single-Document QA | 175 | 97/175 · 55.43% | 108/175 · 61.71% |

## 82 题重跑：按领域

| 领域 | 题数 | Direct 正确 / 全部准确率 | Harness 正确 / 全部准确率 |
|---|---:|---:|---:|
| Code Repository Understanding | 31 | 0/31 · 0.00% | 19/31 · 61.29% |
| Long In-context Learning | 12 | 0/12 · 0.00% | 9/12 · 75.00% |
| Long Structured Data Understanding | 9 | 0/9 · 0.00% | 5/9 · 55.56% |
| Multi-Document QA | 14 | 0/14 · 0.00% | 6/14 · 42.86% |
| Single-Document QA | 16 | 1/16 · 6.25% | 11/16 · 68.75% |

82 题中的 Multi-Document QA 最终为 **6/14（42.86%）**，文档覆盖和 verifier 解析优化未在该领域建立准确率提升。代码仓库理解为 **19/31（61.29%）**，长上下文学习为 **9/12（75.00%）**。这些是子集结果，其难度与长度构成不同于完整数据集。

## 长度计量与“260k+”筛选口径

历史筛选规则为 `estimated_prompt_tokens > 260000 OR cached_context_tokens > 260000`，其中缓存字段混合了两类来源：

- **79/82 题**为 `char_estimate`，即来源字符数除以四。
- **3/82 题**为 `provider_prompt_tokens`，取缓存 Direct usage；其中 **index 64、307** 的 helper 合并了首轮和输出恢复调用的 usage，该值不是单次请求的输入长度。

旧审计把复制后的缓存字段统称为 `provider_prompt_tokens`，即便数值实际来自字符估算。公开报告保留真实计量来源。82 题因此应称为**混合计量口径下的历史 260k+ 筛选集**，不能声称全部 82 题都经过 tokenizer 独立验证且严格超过 262,144-token 路由上限。筛选阈值 260,000 与路由上限 262,144 也不同。

最大保留来源包含 **16,182,936 个字符**；历史 **4,045,734 tokens** 来自**字符数/4 估算**，不是 Qwen 实测 token，不能据此给出经过实测的原生上下文扩展倍数。本实验中，明确的超限拒绝比估算长度更能证明 Direct 路径的容量边界。V3.2 OOLONG 报告中的精确 Qwen token 扩容测量属于另一轮实验。

## 保留配置与评分方式

| 设置 | 完整 503 题 | 专门重跑 82 题 |
|---|---|---|
| 模型 | `Qwen3.8-Flash-Next-FP8` | 同名模型 |
| Harness 路由上下文 / 输出预留 | 262,144 / 8,192 tokens | 相同 |
| 回答上限 / 候选回答上限 | 8,192 / 2,048 tokens | 相同 |
| 来源分块 | 48,000 字符 | 相同 |
| 证据选择 | 8 个窗口 × 4,000 字符，总上限 28,000 字符 | 相同，改进来源覆盖 |
| 候选数 | 3 | 3 |
| Thinking / reasoning effort | 开启 / `medium` | 开启 / `medium` |
| HTTP helper temperature | 0.1 | 0.1 |
| 选项验证 | 早期候选与验证流程 | 开启，上限 4,096 tokens；一次解析重试 |
| 并发 jobs | 4 | 8 |

历史 helper 在最终文本为空时允许一次短的无 thinking 输出恢复请求。完整运行的评分使用 `extract_choice(content) or extract_choice(reasoning)`，采用本地选项解析器。这里明确报告本地评分口径，尚未建立其与官方解析器在原始生成上的严格等价性。[官方结果汇总代码](https://github.com/THUDM/LongBench/blob/main/result.py) 根据传入的 judge 统计准确率，但并不使本地解析成绩自动成为官方提交。

Harness 使用多候选、证据窗口和验证调用；Direct 使用完整提示及可能的输出恢复。本实验没有对两者施加 V3.2 OOLONG 的累计 token 与物理调用次数共同预算合同。保留计数不足以证明同等总计算量比较、完整费用审计或延迟优势；Direct 缓存也早于最后一轮 82 题 Harness 重跑。

## 初期 10 题路由检查

| 方法 | 正确 / 尝试 | 全部准确率 |
|---|---:|---:|
| Direct Qwen | 7/10 | 70.00% |
| 旧 ReaderAgent | 5/10 | 50.00% |
| 初期 Harness | 7/10 | 70.00% |

10 题 Harness 全部进入 `direct_pass` 并复用 Direct 输出，因此验证的是可容纳输入的路由行为，没有测试扩窗后的准确率。该试验严格按最终 content 评分，区别于后续完整运行的 content 或 reasoning 解析，并使用 `xhigh` reasoning effort，不能直接用于跨优化阶段的同口径归因。

## 实验支持的技术价值

历史实验表明：在外部保留完整来源、按窗口选取原始证据，可以为完整提示被直连接口拒绝的任务恢复有效执行。完整 Benchmark **多答对 59 题**，专门重跑**答对 49 道被 Direct 拒绝的题目**。这是已经测得的系统级优势。

当前 V3.2 仍需开展新一轮预先冻结的 LongBench 评测，使用精确的逐请求 token 计量、严格最终答案解析、统一累计预算、独立来源选样和同模型基线。历史成绩不证明当前版本的榜单名次、跨模型家族泛化、同预算超越 RLM/MemGPT 或无限准确上下文。

## 专门重跑 82 题：逐题结果

Index 是保留数据顺序中的零起始 offset；ID 对应原 Benchmark 题目。`—` 表示没有解析出选项，全部题目均保留在准确率分母中。两轮完整派生记录见[结果 JSON](../benchmarks/longbench_v2_historical.json)。

| Index | ID | 标准答案 | Direct | Harness | Harness 路由 |
|---:|---|:---:|:---:|:---:|---|
| 2 | 66f94f9ebb02136c067c4fde | D | — | B | `raw_source` |
| 7 | 66fa208bbb02136c067c5fc1 | B | — | B | `raw_source` |
| 10 | 66ec56dd821e116aacb1cd0e | C | — | D | `raw_source` |
| 12 | 66ebd55f5a08c7b9b35e0698 | A | — | A | `raw_source` |
| 16 | 66f8c9febb02136c067c4511 | C | D | D | `raw_source` |
| 18 | 66f61d7dbb02136c067c1802 | D | — | D | `raw_source` |
| 28 | 66f40b9c821e116aacb30a99 | C | — | D | `raw_source` |
| 31 | 6724631ebb02136c067d7300 | D | — | D | `raw_source` |
| 33 | 66f40e44821e116aacb30b45 | C | — | D | `raw_source` |
| 35 | 66fa3843bb02136c067c655d | D | — | D | `raw_source` |
| 45 | 66ee4c55821e116aacb21596 | D | — | D | `raw_source` |
| 46 | 66faafd9bb02136c067c74a8 | B | — | B | `raw_source` |
| 57 | 67237d8ebb02136c067d6c06 | C | — | C | `raw_source` |
| 58 | 66f1dac1821e116aacb27df1 | A | — | A | `raw_source` |
| 64 | 67246903bb02136c067d74f2 | A | A | A | `direct_pass` |
| 65 | 66f2a4e3821e116aacb2a49b | B | — | B | `raw_source` |
| 74 | 66f2a255821e116aacb2a20b | D | — | D | `raw_source` |
| 96 | 6703f73cbb02136c067cd74a | B | — | D | `raw_source` |
| 115 | 66ebd2475a08c7b9b35e0155 | D | — | A | `raw_source` |
| 116 | 66fa8ccdbb02136c067c6fa7 | B | — | B | `raw_source` |
| 120 | 66f55828821e116aacb3363e | D | — | D | `raw_source` |
| 124 | 66fa2734bb02136c067c627a | A | — | A | `raw_source` |
| 131 | 66f299b3821e116aacb29cfa | B | — | B | `raw_source` |
| 138 | 66f1cd02821e116aacb27d3f | B | — | B | `raw_source` |
| 140 | 66fa542bbb02136c067c686d | A | — | A | `raw_source` |
| 142 | 670765abbb02136c067d06b4 | B | — | B | `raw_source` |
| 143 | 66f3df1e821e116aacb2f7be | A | — | A | `raw_source` |
| 147 | 66ebd34d5a08c7b9b35e035d | C | — | D | `raw_source` |
| 148 | 66ead7bd5a08c7b9b35dca7a | D | — | D | `raw_source` |
| 154 | 6724c5febb02136c067d78e7 | A | — | B | `raw_source` |
| 157 | 66f3fb96821e116aacb3043f | A | — | — | `raw_source` |
| 159 | 66ed5be2821e116aacb1fb57 | D | — | D | `raw_source` |
| 168 | 66f00f1a821e116aacb252cf | B | — | A | `raw_source` |
| 174 | 66f2a46d821e116aacb2a41a | B | — | B | `raw_source` |
| 191 | 66f2cb0f821e116aacb2ba6d | A | — | D | `raw_source` |
| 193 | 66faa0f5bb02136c067c722c | A | — | A | `raw_source` |
| 198 | 66f954a5bb02136c067c511a | B | — | B | `raw_source` |
| 205 | 66f1ae08821e116aacb27c9c | B | — | D | `raw_source` |
| 212 | 66ebcc335a08c7b9b35df8b6 | D | — | D | `raw_source` |
| 217 | 66ebeb5a5a08c7b9b35e1ba5 | A | — | A | `raw_source` |
| 222 | 670cb73fbb02136c067d2502 | C | — | D | `raw_source` |
| 227 | 6704fe26bb02136c067ce670 | C | — | D | `raw_source` |
| 233 | 67040983bb02136c067cd996 | A | — | B | `raw_source` |
| 251 | 66fa1e4cbb02136c067c5edf | A | — | D | `raw_source` |
| 260 | 66fb75adbb02136c067c7d73 | C | — | D | `raw_source` |
| 262 | 6708ae87bb02136c067d1847 | C | — | B | `raw_source` |
| 263 | 66f3fd3a821e116aacb30533 | B | — | B | `raw_source` |
| 264 | 66ebc04a5a08c7b9b35de740 | B | — | D | `raw_source` |
| 283 | 66ed3960821e116aacb1f616 | C | — | C | `raw_source` |
| 297 | 66f52f0f821e116aacb32ddb | A | — | A | `raw_source` |
| 307 | 66f3a311821e116aacb2de19 | B | D | D | `raw_source` |
| 317 | 66f7f382bb02136c067c3a6c | C | — | C | `raw_source` |
| 328 | 66f2a14b821e116aacb2a128 | C | — | C | `raw_source` |
| 331 | 66f43414821e116aacb310a1 | D | — | D | `raw_source` |
| 335 | 6700d19dbb02136c067cb024 | C | — | A | `raw_source` |
| 336 | 66f3ac0b821e116aacb2e203 | D | — | D | `raw_source` |
| 348 | 66fa69a4bb02136c067c6b75 | D | — | D | `raw_source` |
| 350 | 66fa6702bb02136c067c6abb | B | — | B | `raw_source` |
| 357 | 6707f349bb02136c067d13b9 | C | — | C | `raw_source` |
| 368 | 66fb77e7bb02136c067c7db1 | C | — | D | `raw_source` |
| 369 | 66fa5fbcbb02136c067c6959 | B | — | A | `raw_source` |
| 371 | 66f3e318821e116aacb2f9d1 | B | — | D | `raw_source` |
| 372 | 66fa1b54bb02136c067c5df7 | B | — | B | `raw_source` |
| 390 | 66f2982b821e116aacb29c66 | A | — | A | `raw_source` |
| 394 | 66f11349821e116aacb269e0 | A | — | C | `raw_source` |
| 395 | 66f28794821e116aacb298ee | A | — | A | `raw_source` |
| 405 | 66f3c5f2821e116aacb2eccf | A | — | C | `raw_source` |
| 407 | 67066b87bb02136c067cfaa3 | B | — | A | `raw_source` |
| 410 | 66f24538821e116aacb2865e | D | — | C | `raw_source` |
| 415 | 66f57379821e116aacb33b7a | D | — | D | `raw_source` |
| 423 | 66f3e604821e116aacb2fadf | D | — | D | `raw_source` |
| 427 | 6700eb6ebb02136c067cb152 | D | — | D | `raw_source` |
| 438 | 66ec17e4821e116aacb1a6a7 | D | — | D | `raw_source` |
| 447 | 66fcf80dbb02136c067c928e | B | — | B | `raw_source` |
| 449 | 6724cae7bb02136c067d79be | B | — | D | `raw_source` |
| 453 | 66f40c5e821e116aacb30ad2 | A | — | B | `raw_source` |
| 455 | 66f51ab2821e116aacb325fb | C | — | C | `raw_source` |
| 462 | 66f908e3bb02136c067c4992 | A | — | C | `raw_source` |
| 468 | 67246d0ebb02136c067d75f0 | D | — | D | `raw_source` |
| 470 | 66ed24c5821e116aacb1eee6 | B | — | B | `raw_source` |
| 471 | 66ebdd1c5a08c7b9b35e113b | D | — | D | `raw_source` |
| 486 | 66fa36c9bb02136c067c6517 | C | — | C | `raw_source` |
