# Context Is All You Need

## Context Read and Understand Harness (CRU-Harness) · V3.2

[English](README.md) | **简体中文**

**让有限的模型窗口，承载百万级来源上的证据工作。**

**CRU-Harness** 是面向超长材料的证据计算执行系统。V3.2 由软件持有和索引完整来源，模型在有界工作窗口内完成必要的语义判断，来源事实持续累积，统计与条件聚合由本地程序执行。无需修改模型权重或增加训练，即可围绕超过单次输入容量的材料继续处理任务。

**容量突破发生在系统执行层，底座模型的单次注意力窗口保持不变。** 项目的革命性价值在于把上下文组织成可访问、可累积、可复用、可计算的软件资源，使材料规模、当前推断窗口与已取得的事实可以分别管理。

[技术实现](docs/v32_implementation.md) · [评测与逐题结果](docs/benchmark_v32.md) · [结果 JSON](benchmarks/oolong_v32_20261101.json) · [完整项目书](docs/CRU-Harness_Project_Proposal_V3.2.docx) · [执行 API](docs/harness_api.md)

## 最新成果

以下数据来自 2026-11-01 冻结的 V3.2 OOLONG-synth 本地受控实验，本次独立发布沿用该实验记录。12 个来源具有不同的 context ID 和内容哈希，未参与本项目此前调优；实测长度全部超过本次 Qwen 服务报告的原生 **262,144 tokens** 上限。

| 指标 | V3.2 实测结果 | 含义 |
|---|---:|---|
| 最大来源规模 | **1,866,427 Qwen tokens** | 原生服务上限的 **7.12 倍** |
| 最长获满分来源 | **1,792,309 Qwen tokens** | 在原生长度的 **6.84 倍**来源上获官方满分 |
| 超原生来源满分率 | **9/12，75.00%** | 全部 12 个来源均超过原生服务上限 |
| Benchmark 官方均分 | **81.25/100** | 包含数值题的部分分，不等同于准确率 |
| 平均每题模型用量 | **25,859 tokens** | 包含输入与输出；较同批适配 RLM 少 **27.74%** |

最高来源长度与最长满分来源分别报告，避免把“材料进入执行流程”与“答案正确”混为一谈。这些数字衡量系统可处理的来源空间，不表示模型一次读取了全部正文。

### 基准来源与评测协议

本项目使用已有公开 **OOLONG-synth** 基准中来源不重叠的子集。基准由论文 [*Oolong: Evaluating Long Context Reasoning and Aggregation Capabilities*](https://arxiv.org/abs/2511.02817) 提出，上游来源为[官方代码仓库](https://github.com/abertsch72/oolong)和[官方 Hugging Face 数据集](https://huggingface.co/datasets/oolongbench/oolong-synth)。

| 项目 | 冻结评测协议 |
|---|---|
| 数据集 | `oolongbench/oolong-synth`，官方 `test` split |
| 数据 revision | `81d2f8d9349f6ff329d695f9be494051e9f05cb9` |
| 官方 scorer commit | `0bb7eabe839218fee7fe8d007f41cfc2fd3ae24c` |
| 评测单位 | **12 个不同来源上下文，每份来源只选一题** |
| 名义长度分层 | 1,048,576 tokens 层 6 份，2,097,152 tokens 层 6 份 |
| Qwen 实测来源长度 | **745,202–1,866,427 tokens**，对完整原始来源计量 |
| 留出排除 | 排除此前使用的 56 个上下文，**context ID 与全文哈希重叠均为零** |
| 随机种子 | **样本选择：20261105**；**模型生成：20261101** |

数据集名义长度与 Qwen tokenizer 实测长度是两种计量。选样仅用元数据，问题与答案未用于选样或当前版本调优。来源上下文不重叠不代表底层记录相互独立，也不能排除底座模型预训练接触。本轮为本地封存评测，不是官方线上排行榜成绩。

**官方分**包含数值题部分分（`0.75 ** abs(error)`）；**满分**指官方分等于 1；**解析后精确匹配（EM）**指经官方答案解析器处理后的字符串完全一致。因此 CRU-Harness 分别报告 **81.25/100 官方均分**、**9/12 满分**与 **4/12 解析后 EM**。[公开结果 JSON](benchmarks/oolong_v32_20261101.json) 包含固定的数据/scorer/基线版本、关键源码 SHA-256、各上下文来源哈希以及逐题评分与用量；原文与私人请求日志留在本地。

### 同模型、同预算三路对比

模型为 `Qwen3.8-Flash-Next-FP8`，temperature=0、seed=20261101、关闭 thinking。每题每方法的共同上限为：单次输入加输出 **16,384 tokens**，累计 **65,536 tokens**，最多 **32 次物理生成**，每次最多 **2,048 输出 tokens**。

| 方法 | 满分题 / 满分率 | 官方均分 /100 | 平均输入＋输出 tokens |
|---|---:|---:|---:|
| **CRU-Harness V3.2（`adaptive_v3`）** | **9/12 · 75.00%** | **81.25** | **25,859** |
| RLM 官方引擎＋环境适配 | 7/12 · 58.33% | 58.33 | 35,788 |
| MemGPT 原控制器＋I/O 适配 | 0/12 · 0.00% | 0.00 | 55,586 |

同预算指共同资源上限，实际消耗分别计量。所有预算耗尽、未知答案和无响应都保留在分母中。RLM 有 5 条预算失败；MemGPT 有 7 条预算失败，仅 1 条交付非空答案，因而不能把这一配置的结果外推为原论文方法的普遍能力。适配细节、置信区间、完整对照与逐题结果见[审计报告](docs/benchmark_v32.md)。

本轮是 12 个上下文、每题一次推理的先导实验，不是官方排行榜提交。公开数据的预训练污染和底层记录重叠未被排除，当前证据不支持宣称普遍或显著领先。

### OpenAI 三模型迁移重放 · 2026-11-02

保持 V3.2 算法不变，通过官方 Responses API，在**同一组 12 个 OOLONG-synth 来源**上运行以下三个精确模型。这检验了向另一提供方迁移的表现；来源是已用评测集，每模型每来源仅运行一次，本轮不依据新输出调优。

| 模型 | 满分题数／满分率 | 官方均分／100 | 平均输入＋输出 tokens | 执行错误 |
|---|---:|---:|---:|---:|
| `gpt-6-sol` | **9/12 · 75.00%** | **75.00** | 25,471 | 0 |
| `gpt-6-luna` | **6/12 · 50.00%** | **50.00** | 19,400 | 3 |
| `gpt-6.1-terra` | **10/12 · 83.33%** | **83.33** | 25,441 | 0 |

全部 36 项结果保留在分母中。Luna 的三项错误均为触及输出上限的已接收响应，usage 已知。三路统一使用 reasoning=none、单次 16,384 和每题累计 65,536 token 上限，不发送 temperature 或 seed。319 次生成共使用 **843,748 tokens**，保守生成费用估算为 **$4.699397**，不是实际账单。

完整来源输入封装计数为 **651,081–1,582,486 OpenAI input tokens**。计入完整 direct 提示和输出预留后，6 个来源超过公开的 1,050,000-token 原生窗口；该子组满分题数为 **Sol 4/6、Luna 2/6、Terra 5/6**。这种计量不同于 Qwen 原文 token 数。本轮 direct 只计数、没有生成比较臂，因此不能估计对底座的因果增益。结果支持这组任务上的迁移，同时暴露了 Luna 的输出协议兼容性限制。

[完整协议与逐题结果](docs/openai_transfer_20261102.zh-CN.md) · [English](docs/openai_transfer_20261102.md) · [公开结果 JSON](benchmarks/openai_transfer_20261102.json)

无需凭证或网络即可核对公开指标：

```bash
python3 -S scripts/verify_openai_transfer_results.py
```

### LongBench v2：全量成绩与超长上下文恢复

早期证据窗口版 Harness 还在 **LongBench v2** 上完成了评测，来源为[官方论文](https://arxiv.org/abs/2412.15204)、[官方仓库](https://github.com/THUDM/LongBench)和 [503 题公开数据集](https://huggingface.co/datasets/zai-org/LongBench-v2)。两条路径均使用 `Qwen3.8-Flash-Next-FP8`。

| 实验记录 | Direct Qwen 正确 / 全部 | Harness 正确 / 全部 | Harness 有答案 |
|---|---:|---:|---:|
| 历史 503 题全量运行 | 254/503 · 50.50% | **313/503 · 62.23%** | 502/503 |
| 最后统一重跑的 82 题超长子集 | 1/82 · 1.22% | **50/82 · 60.98%** | 81/82 |

全量结果提升 **11.73 个百分点**。在 82 题记录中，Direct 有 **79 题因上下文长度被服务拒绝**，Harness 在这些相同题目上答对 **49/79（62.03%）**，体现了超出服务输入容量后恢复有效解题路径的价值。容量拒绝与模型答错分别记录；两条路径的推理计算量不同，Direct 使用已有缓存。

这些是**历史证据窗口路径的本地评测**，公开样本已经过多轮开发使用。82 题筛选混合字符估算和缓存 usage，不代表全部经过 tokenizer 实测超出原生上限；它与 503 全量属于不同运行阶段。两份成绩均不是当前 V3.2 结构化事实版本的新重跑，也不是官方排行榜提交。详见[完整中文报告、领域成绩和 82 题逐题表](docs/benchmark_longbench_v2.zh-CN.md)、[English report](docs/benchmark_longbench_v2.md)及[全部逐题结果与来源哈希](benchmarks/longbench_v2_historical.json)。

无需模型调用，即可在本地复算发布结果：

```bash
python scripts/verify_longbench_results.py
```

## 相似工作对比

CRU-Harness 把外部上下文管理与面向任务的证据计算结合起来。下表比较架构取向；上方实测三路表使用固定版本、同模型的本地适配配置，不混排论文中异模型、异预算的分数。

| 方法 | 上下文组织 | 聚合方式 | 来源与不确定性 | 预算取向 |
|---|---|---|---|---|
| **RAG（原始论文）** | 从外部索引检索片段，为生成提供条件 | 模型综合检索证据，也可扩展程序算子 | 检索片段提供证据，全文覆盖与记录边界取决于应用实现 | 检索规模与生成窗口 |
| **RLM** | 通过持久 REPL 访问外部上下文，递归调用模型 | 可编程代码、归约与子模型调用 | 可检查或记录 REPL 状态与执行轨迹，也可编程实现来源追踪和边界 | 递归执行及可配置的深度、调用和 token 限制 |
| **MemGPT（原始架构）** | 通过工具与记忆换页管理 core、recall、archive 记忆 | 对文档进行多步检索、阅读与推理 | 检索文档支持基于证据的回答，类型化记录账本不是其核心抽象 | 有限活动上下文与记忆管理 |
| **CRU-Harness V3.2** | 来源索引、有界模型工作空间与持续来源事实 | 类型化本地算子、完整正文等价复用、确定性账本结算 | 显式记录身份、跨度、哈希、重数、冲突、未知与分组边界 | 单次请求、累计 token 与物理生成共享上限，并可重放计量 |

一手来源：RAG [论文](https://arxiv.org/abs/2005.11401) / [作者指向的 Transformers 实现](https://github.com/huggingface/transformers/tree/main/src/transformers/models/rag)；RLM [论文](https://arxiv.org/abs/2512.24601v3) / [代码](https://github.com/alexzhang13/rlm)；MemGPT [论文](https://arxiv.org/abs/2310.08560) / [历史官方仓库](https://github.com/letta-ai/letta)。

本轮 MemGPT 实验指公开 JSON 中固定的历史原控制器。RLM 已支持可编程聚合和持久 REPL 执行，MemGPT 已支持多步文档检索与证据使用。CRU-Harness 的贡献在于把**来源绑定的类型化事实、完整正文复用、确定性聚合、显式未知边界与共享预算审计**整合为同一执行合同；不宣称外部记忆、检索或代码聚合为本项目独有。详见[相似工作完整对比](docs/related_work.md)。

## 核心技术优势

### 1. 来源空间与模型工作空间解耦

完整提示可容纳时直接交给模型；超预算时使用来源索引、明确范围与局部证据。模型窗口承担当前推断，完整材料和执行状态由外部系统保留，来源增大不再自动要求单次请求同步增大。

### 2. 语义判断与类型化计算协作

查询计划分别表达范围、分类字段、语义条件、分组键和归约方式。模型产生原子标签，本地执行计数、频率比较、分组统计、日期极值与数值聚合。`conditional_group` 支持 **标签筛选 → 按用户分组 → 计数 → argmax/argmin**，避免把语义类别与用户身份混作同一维度。

### 3. 来源绑定的事实持续累积

短 alias 与真实记录、源文跨度、原始哈希的映射留在本地。通过校验的分类进入 `FactLedger`；同一物理记录重复提交不重复计数，不同物理实例保留各自重数；冲突和未知保持显式状态。结构化答案从完整账本结算，不依赖最后一次提示能装下多少历史窗口。

### 4. 完整正文等价复用

在标签仅由当前正文决定的任务中，完全相同的投影正文只分类一个代表，再按各物理实例的来源、用户、日期和真实重数回填。最新 V3.2 运行中，**47,202 条等价事实转移**得到独立核验；一个案例以 **533 种正文**覆盖 **39,056 条物理记录**。

复用要求完整字符串相等，不做模糊语义合并。模型真正阅读的范围与派生覆盖范围分别记录；代表分类与实际标注的一致性仍需验证。

### 5. 未知量边界与有限追检

各组保留已知匹配数、未知量及上下界。只有一个候选下界严格超过全部竞争者上界，才允许提前确定赢家；并列保留集合。每条记录与同文组共享尝试上限，连续无事实增益即停止，防止通过重复副本绕过限制。

### 6. 统一预算与可重放审计

分类、计划修复、递归、摘要和最终回答共享输入输出 token 与物理生成账本。发送前用服务端 tokenizer 计量并预留输出；原始返回、来源绑定、事实转移和最终状态可独立重放。

V3.2 的 **12/12 份机制结果通过检查**，100 次原始分类响应、5,931 个正文绑定与 52,839 条保留来源记录得到核验。整体实验包含对照的 **24/24 份 Harness 机制检查通过**，官方评分复算零差异。最新实验前的 **372 项相关检查通过**；当前 checkout 下方命令覆盖的 **76 项核心检查通过**，两者验收范围不同。

## 架构

```mermaid
flowchart TD
    S["完整来源与任务"] --> H["预算路由"]
    H -->|"完整提示可容纳"| D["模型直接回答"]
    H -->|"超出请求预算"| I["来源索引与范围选择"]
    I --> P["按需语义分类与事实获取"]
    P -->|"来源绑定与正文等价转移"| F["持续事实状态"]
    I -->|"明确源字段"| O["类型化聚合与未知量边界"]
    F --> O
    O --> A["答案与执行审计"]
    D --> A
    B["统一生成预算账本"] -.-> H
    B -.-> P
```

| 模块 | 作用 |
|---|---|
| [`harness.py`](long_context_agent/harness.py) | 模型优先路由与执行入口 |
| [`adaptive_evidence_v3.py`](long_context_agent/adaptive_evidence_v3.py) | 查询计划、分类获取、追检、停止与最终结算 |
| [`fact_ledger.py`](long_context_agent/fact_ledger.py) | 来源事实、冲突/未知、聚合及分组边界 |
| [`record_protocol.py`](long_context_agent/record_protocol.py) | 完整正文投影、短 alias 协议和严格解码 |
| [`semantic_coverage.py`](long_context_agent/semantic_coverage.py) | 完整正文等价索引与来源绑定转移 |
| [`experiment_budget.py`](long_context_agent/experiment_budget.py) | 真实 tokenizer 准入与共同资源账本 |
| [`independent_ingestion/`](independent_ingestion/) | 可替换解析器与统一 DocumentIR |
| [`reading_memory/`](reading_memory/) | 阅读锚点、内部记忆和本地状态存储 |

## 安装与快速试用

项目以 CRU-Harness 独立发布；`long_context_agent` 等名称是保留的 Python 导入包名。

推荐 Python 3.12，独立核心支持 Python 3.10+。在仓库根目录运行。以下虚拟环境命令适用于 macOS / Linux；核心离线演示仅使用 Python 标准库。

```bash
git clone https://github.com/quxiaoqiya/CRU-Harness.git
cd CRU-Harness
python3 -m venv .venv
source .venv/bin/activate
python examples/demo_harness.py
```

### 无密钥、无网络的机制演示

[`examples/demo_harness.py`](examples/demo_harness.py) 使用**模拟模型响应与模拟 token 计数**，构造 300 条、2 种重复正文，调用实际 V3.2 API 并检查来源事实和确定性聚合。典型输出包含：

```text
mode: adaptive_fact_aggregation_v3
answer: Answer: 200
source_records: 300
unique_bodies: 2
directly_classified_records: 2
transferred_records: 298
known_records: 300
unknown_records: 0
final_fact_loss_count: 0
```

这是执行机制演示，**不是 Benchmark**，不能证明真实模型的准确率或 token 节省。

### 接入真实模型服务

在线调用使用 `LongContextHarness.answer_with_structured_facts(sample, client)`。`ProviderTransport` 要求 OpenAI 风格 Chat Completions 和精确 `/tokenize` 计量。官方 OpenAI Responses API 使用下方独立适配器。两条路径都要求精确计数和完整 usage。

先在本地环境中配置 `Q38_BASE_URL`、`Q38_MODEL`、`Q38_API_KEY`，再准备 UTF-8 文本 `context.txt`：

```python
import os
from pathlib import Path

from long_context_agent import LongContextHarness
from long_context_agent.experiment_budget import (
    BudgetedClient, ExperimentBudget, ProviderTransport,
)

transport = ProviderTransport(
    base_url=os.environ["Q38_BASE_URL"],
    model=os.environ["Q38_MODEL"],
    api_key=os.environ["Q38_API_KEY"],
)
budget = ExperimentBudget(
    context_tokens=16_384, total_tokens=65_536,
    max_calls=32, max_output_tokens=2_048,
)
client = BudgetedClient(transport, budget)
harness = LongContextHarness(
    None, max_context_tokens=budget.context_tokens,
    reserve_tokens=budget.max_output_tokens,
)
result = harness.answer_with_structured_facts({
    "id": "local-example",
    "context": Path("context.txt").read_text(encoding="utf-8"),
    "question": "根据来源回答问题，并给出所需的简洁结果。",
}, client)
print(result["answer"])
print(client.summary())
```

将问题替换为实际任务。短文本可能进入 `direct_pass_v3`；超过请求预算时才进入记录聚合或事实读取路径。服务端需返回完整 `usage`；模型输出和工具 schema 同样计入共同预算。请勿提交凭据、私人来源或运行缓存。

### OpenAI Responses 适配器

沿用相同的 `budget`、`harness` 和输入对象，替换 transport 与 client：

```python
from long_context_agent.openai_responses_transport import (
    OpenAIResponsesTransport, OpenAIResponsesBudgetedClient,
)

transport = OpenAIResponsesTransport(
    api_key=os.environ["OPENAI_API_KEY"],
    model="gpt-6-sol",
    reasoning_effort="none",
)
client = OpenAIResponsesBudgetedClient(transport, budget)
```

适配器使用 Responses 输入 token 计数，关闭自动截断和存储，请求默认服务档，并将输入、输出用量纳入预算。不发送旧采样字段或 chat 模板参数，不做隐式重试，未完成响应会明确记录。当前仅支持文本，不支持模型托管工具。

### 本地文档接入与核心检查

```bash
python -m independent_ingestion.cli paper.md --json-out work/demo/document.ir.json

# 可选：PDF 文本解析
python -m pip install pypdf

# 无模型网络调用的核心回归
python -m pip install -r requirements-dev.txt
python -m pytest -q \
  tests/test_adaptive_evidence_v3.py \
  tests/test_semantic_coverage_v32.py \
  tests/test_record_protocol.py \
  tests/test_experiment_budget.py
```

本仓库的独立 Harness 离线演示仅需标准库；开发测试依赖列在 `requirements-dev.txt`。真实科研文献抽取、复杂 PDF/视觉解析与数据库发布需独立验收；此处不宣称相关生产链已经完整交付。

## 文档与验证边界

- [V3.2 技术实现](docs/v32_implementation.md)
- [最新 OOLONG 评测与逐题结果](docs/benchmark_v32.md)
- [机器可读结果 JSON](benchmarks/oolong_v32_20261101.json)
- [OpenAI 三模型迁移评测](docs/openai_transfer_20261102.zh-CN.md)
- [OpenAI 迁移结果 JSON](benchmarks/openai_transfer_20261102.json)
- [LongBench v2 成绩与评测协议](docs/benchmark_longbench_v2.zh-CN.md)
- [LongBench v2 逐题结果 JSON](benchmarks/longbench_v2_historical.json)
- [完整项目书](docs/CRU-Harness_Project_Proposal_V3.2.docx)
- [执行 API 与预算规则](docs/harness_api.md)
- [DocumentIR 合同](docs/document_ir_contract.md)
- [模型能力接口](docs/model_capability_gateway.md)
- [内部记忆与产品投影边界](docs/product_projection_boundary.md)
- [数据使用说明](DATA_LICENSE.md)

来源绑定、计算闭合与语义正确性分别验收：来源跨度有效证明材料来自输入，确定性聚合证明账本结算一致，模型分类与真实标注是否一致仍是独立问题。当前支持的是可扩展、可追溯的执行能力，不是无限准确上下文，也不代表所有超长任务都能在固定预算内解决。

## 发布范围与许可

仓库交付独立 Harness、输入解析与记忆接口、可运行示例、相关测试和最新评测摘要。原始语料、私人部署配置、模型请求日志和运行环境留在本地。

当前未为项目原创源码指定开源许可证；第三方代码遵循各自许可，随附 OOLONG 评分夹具的 MIT 通知见 [`third_party/oolong/LICENSE`](third_party/oolong/LICENSE)。公开原文数据的分发仍受其原始来源条款约束。
