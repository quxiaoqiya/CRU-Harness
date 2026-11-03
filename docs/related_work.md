# Benchmark 来源与相关工作 / Benchmark provenance and related work

核验日期 / Verified: 2026-11-02。项目名称为 **Context Read and Understand Harness（CRU-Harness）**；下面的 OOLONG、RAG、RLM 和 MemGPT 均为各自作者的既有工作。

## English overview

CRU-Harness combines typed query plans, source-bound fact accumulation, exact-body reuse that preserves physical record multiplicity, deterministic aggregation, explicit unknown bounds, and shared-budget auditing. External knowledge, memory management, recursive reading, executable aggregation, and execution logs already appear in prior work; they are not claimed as CRU inventions. The comparison below identifies a concrete implementation focus rather than an exclusive first claim. See the original [RAG paper](https://arxiv.org/abs/2005.11401), [RLM paper](https://arxiv.org/abs/2512.24601v3), and [MemGPT paper](https://arxiv.org/abs/2310.08560).

The latest local pilot uses 12 new OOLONG-synth source contexts with the same Qwen model and common generation-budget ceilings. CRU-Harness scores 81.25/100 versus 58.33 for the pinned RLM adaptation, with 27.74% fewer charged model tokens. Full credit is 9/12 versus 7/12. These are local adapted results, not the papers’ original scores or an official leaderboard submission. The paired full-credit McNemar p-value is 0.625; the small sample does not establish a stable general advantage. [Results and limitations](benchmark_v32.md) · [Machine-readable results](../benchmarks/oolong_v32_20261101.json).

## 1. Benchmark 由谁提出，当前实验使用了什么

**Oolong: Evaluating Long Context Reasoning and Aggregation Capabilities** 由 **Amanda Bertsch、Adithya Pratapa、Teruko Mitamura、Graham Neubig、Matthew R. Gormley** 提出。它评估文本单元的原子分析与聚合，包括分类、计数以及时间和用户关系；论文区分 Oolong-synth 与 Oolong-real。CRU-Harness 使用其中的 OOLONG-synth，不能把这个公开 benchmark 描述成本项目原创数据集。[官方论文](https://arxiv.org/abs/2511.02817)

| 来源 | 官方入口 | 与本轮实验的关系 |
|---|---|---|
| 论文 | [arXiv:2511.02817](https://arxiv.org/abs/2511.02817) | Benchmark 定义与原作者；与本地 CRU 实验是不同成果。 |
| 官方代码与评分实现 | [abertsch72/oolong](https://github.com/abertsch72/oolong) | 提供上游评测代码。 |
| 官方数据集 | [oolongbench/oolong-synth](https://huggingface.co/datasets/oolongbench/oolong-synth) | 本轮从 test split 选择新来源；数据卡关联上述论文。 |
| 数据仓库固定 revision | [81d2f8d9349f6ff329d695f9be494051e9f05cb9](https://huggingface.co/datasets/oolongbench/oolong-synth/commit/81d2f8d9349f6ff329d695f9be494051e9f05cb9) | 与公开结果 JSON 一致；官方 [Files](https://huggingface.co/datasets/oolongbench/oolong-synth/tree/main) 和 [Commit History](https://huggingface.co/datasets/oolongbench/oolong-synth/commits/main) 可定位该提交。 |
| 下载数据的 Parquet revision | `d0b8c03122f8a54aced6009cb410b10a0120ea61` | 另行保留实际数据文件版本；不要与数据仓库 revision 互换。 |
| 官方 scorer 固定 commit | [0bb7eabe839218fee7fe8d007f41cfc2fd3ae24c](https://github.com/abertsch72/oolong/tree/0bb7eabe839218fee7fe8d007f41cfc2fd3ae24c) | 本轮采用该版本的 `src/eval/eval_helpers.py`。 |
| 固定 scorer 文件 | [上游源码](https://raw.githubusercontent.com/abertsch72/oolong/0bb7eabe839218fee7fe8d007f41cfc2fd3ae24c/src/eval/eval_helpers.py) · [本仓库副本](../third_party/oolong/eval_helpers_original.py) | 上游数值部分分为 `0.75 ** abs(error)`；因此官方均分与满分率分别报告。 |

版本和来源哈希保存在[公开结果 JSON](../benchmarks/oolong_v32_20261101.json)，其关键源码指纹、样本 ID 和内部审计摘要仍沿用原封存身份。更换展示品牌不改动这些实验标识。第三方数据和代码按各自来源条款使用；它们的许可不自动成为 CRU-Harness 原创源码的许可。

这批实验是公共 benchmark 上的**本地受控先导评测**，不是 OOLONG 官方发布的模型成绩，也没有提交官方在线排行榜。样本来源不重叠不等于底层记录完全独立，公开数据的预训练接触也不能排除。[完整评测说明](benchmark_v32.md)

## 2. 相关工作按机制比较

| 技术路线 | 既有工作已经具备的能力 | CRU-Harness 的具体侧重点 | 共同点与比较边界 |
|---|---|---|---|
| 检索增强生成（RAG） | 原论文把参数化生成模型与外部密集索引、神经检索器结合，基于检索结果生成答案。[论文](https://arxiv.org/abs/2005.11401) | 对记录任务显式表达范围、语义条件、分组与归约，将接受的语义标签累积为来源事实后进行确定性结算。 | 外部知识访问和来源可检查性已有先例。“RAG”现已是广泛方法族，不能概括为都没有状态或都不能做聚合。本轮没有 RAG 比较臂。 |
| 分层记忆与虚拟上下文（MemGPT） | 主上下文与外部记忆之间换入换出，使用函数和控制流管理记忆；论文覆盖长文分析与跨会话任务。[论文](https://arxiv.org/abs/2310.08560) | 为统计建立记录身份、标签、物理重数、未知量与结算算子，单独检查事实保留和聚合过程。 | 外部记忆、多步取证与文档依据不是本项目首创。本轮使用固定历史控制器的适配配置，不代表当前 Letta 平台整体能力。 |
| 外部环境与递归程序执行（RLM） | 把长输入放入外部环境，模型通过程序检查、分解和递归子调用处理材料；代码可以执行聚合。官方实现也有持续 REPL 与轨迹日志。[论文](https://arxiv.org/abs/2512.24601v3) · [官方实现](https://github.com/alexzhang13/rlm) | 用受约束的类型化计划和来源账本组织记录计算，复用完全相同正文的判定并保留每个实例的重数，以未知上下界检查结论是否闭合。 | 超窗口处理、程序化聚合、持久状态和日志均不能作为排他首创。通用 RLM 也可表达类似计算；本项目价值在于已落实的合同、审计与当前受控表现。 |
| CRU-Harness V3.2 | 以现有模型承担局部语义判断，以本地状态与程序组织跨窗口任务。 | 把来源协议、类型化计划、事实保留、等价复用、未知边界、预算与独立重放组合为可检查的执行链。 | 当前实验证据来自记录型文本任务。来源覆盖和算术一致不保证标签语义正确，固定预算也不保证所有超长任务收敛。 |

表中描述原论文及核验时的官方实现。官方仓库会继续演进；上表提及的功能不意味着本轮固定 commit 或受限适配启用了所有当前功能。RAG 论文介绍的是特定模型与训练方案，不能用它替代对所有现代检索系统的定义。

## 3. 这项组合带来的可验证优势

**语义判断和计算规则分别检查。** 查询计划明确范围、分类字段、条件、分组键与归约算子，模型标签经 alias、记录身份和来源位置校验进入账本。程序随后结算已接纳事实，避免把全部统计重新交给最终提示中的自然语言推理。实现见[类型化执行器](../long_context_agent/adaptive_evidence_v3.py)、[记录协议](../long_context_agent/record_protocol.py)和[事实账本](../long_context_agent/fact_ledger.py)。

**重复正文复用而不丢失真实重数。** 在标签仅由完整正文决定的当前查询中，严格相同正文可以共享一个代表判定，回填时仍保留每个物理实例的用户、日期、来源和计数贡献。它降低重复判定需求，但代表误判也可能随重数放大。高重复案例的约 73 倍正文复用潜力不是实测加速，更不是模型注意力窗口扩大。[正文等价覆盖实现](../long_context_agent/semantic_coverage.py) · [评测与失败分析](benchmark_v32.md)

**未知量、预算和事实损失可以分别审计。** 未知/冲突不静默变成负例；分组计数保留上下界，所有模型生成共用预算账本，最终结算基于保留状态。最新运行没有发现已接纳事实的最终丢失，但仍存在语义计数错误、分组预算不足和关键 null 未裁决。这里证明的是执行过程可检查，不能把它称为语义答案的形式化正确性证明。[统一预算客户端](../long_context_agent/experiment_budget.py) · [公开执行摘要](../benchmarks/oolong_v32_20261101.json)

这三点是 CRU-Harness 的具体工程与方法贡献。当前没有“该组合在全部文献中首次出现”的独立检索或理论证明，也没有逐个组件的消融证据足以把整体成绩提升全部归因给某一个机制。

## 4. 当前实测比较支持到哪里

以下数值只取最新 12 个新来源、同一 Qwen3.8-Flash-Next-FP8 和共同预算上限的实验；不混排原论文分数。[公开 JSON](../benchmarks/oolong_v32_20261101.json)

| 本轮配置 | 满分题 / 满分率 | 官方均分 / 100 | 平均输入输出 tokens |
|---|---:|---:|---:|
| CRU-Harness V3.2（公开摘要键 `harness_v32`，原运行键 `adaptive_v3`） | 9/12，75.00% | 81.25 | 25,859 |
| RLM 官方引擎加本地适配 | 7/12，58.33% | 58.33 | 35,788 |
| MemGPT 原控制器加本地适配 | 0/12，0.00% | 0.00 | 55,586 |

相对本轮适配 RLM，官方均分高 **22.92 分**，平均模型 tokens 少 **27.74%**。来源最大为服务报告原生窗口的 **7.12 倍**，最长满分来源为 **6.84 倍**；后者通过元数据统计完成，不证明模型理解了全部正文。这些结果支持当前任务、版本与预算下的数值优势。[容量与成本口径](benchmark_v32.md)

完整比较仍受下列条件约束：

- 样本只有 12 个，每题每方法推理一次；满分配对检验 **p=0.625**，均分差的配对 bootstrap 95% 区间为 **−2.08 至 50.00 分**，不能宣称稳定、普遍或统计显著超越 RLM。
- 同预算指相同上限：每次输入加输出最多 16,384 tokens，每题累计最多 65,536 tokens、最多 32 次生成；实际消耗分别记账，不是相同 FLOPs。
- RLM 采用受限本地 REPL、共享预算与最大深度 2；MemGPT 改用 BM25/本地存储及原生工具适配。这些选择影响交付表现，不能把本表写成原论文模型或当前产品的 SOTA 对比。
- MemGPT 的 0 分包含 7 次预算耗尽、4 次空输出和 1 次未得分的非空输出。所有失败保留在分母；该结果不证明分层记忆路线无效。
- 3 题满分来自未完整覆盖的相对频率估计；全覆盖计数也有错误。现有审计支持来源保留和结算一致性，尚未逐记录独立验证模型标签真值。
- 本轮没有 direct Qwen 或 RAG 比较臂，也没有证实相对 RLM 的时延优势。跨模型、更多任务和组件消融仍需新实验。

## 5. 原论文与官方代码入口

| 工作 | 论文与作者 | 官方代码 / 实现入口 |
|---|---|---|
| Oolong | Amanda Bertsch、Adithya Pratapa、Teruko Mitamura、Graham Neubig、Matthew R. Gormley；[Oolong: Evaluating Long Context Reasoning and Aggregation Capabilities](https://arxiv.org/abs/2511.02817) | [abertsch72/oolong](https://github.com/abertsch72/oolong) · [官方 OOLONG-synth 数据](https://huggingface.co/datasets/oolongbench/oolong-synth) |
| RAG | Patrick Lewis 等；[Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) | 论文正文脚注说明实验代码已进入 HuggingFace Transformers；[当前 RAG 源码目录](https://github.com/huggingface/transformers/tree/main/src/transformers/models/rag) · [官方模型文档](https://huggingface.co/docs/transformers/model_doc/rag) |
| RLM | Alex L. Zhang、Tim Kraska、Omar Khattab；[Recursive Language Models，v3](https://arxiv.org/abs/2512.24601v3) | [alexzhang13/rlm](https://github.com/alexzhang13/rlm)；本轮固定 commit 为 `d04208afbad29ca675ab13478c40ee8bebc84bfe` |
| MemGPT | Charles Packer 等；[MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560) | [历史官方仓库 Letta](https://github.com/letta-ai/letta) · [本轮固定原控制器源码](https://raw.githubusercontent.com/letta-ai/letta/a0139b5fdc203f827168cb741c9f4fea3cec09bf/memgpt/agent.py) |

核验时 Letta 官方仓库把活跃开发指向 [letta-ai/letta-code](https://github.com/letta-ai/letta-code)，并保留历史来源。CRU 本轮固定的原 MemGPT 控制器 commit 为 `a0139b5fdc203f827168cb741c9f4fea3cec09bf`；本地适配没有评测当前 Letta Code 产品。[官方仓库说明](https://github.com/letta-ai/letta)

原方法与数据的学术归属保留给各自作者；CRU-Harness 是本仓库执行系统的独立品牌。

