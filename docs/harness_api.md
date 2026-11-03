# CRU-Harness V3.2 API

当前主要入口为 `LongContextHarness.answer_with_structured_facts(sample, client, config=None)`。
它执行真实任务：完整提示可容纳时直通模型，超预算时进入记录聚合或原文事实读取。

## 输入和资源合同

`sample` 包含 `context`、`question` 两个字符串，可附 `id`。不要把标准答案或评分函数传入执行器。
每个任务创建独立 `BudgetedClient`，分类、修复和最终生成共用其输入输出 token 与请求账本。

```python
import os
from pathlib import Path
from long_context_agent import LongContextHarness, AdaptiveV3Config
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
result = harness.answer_with_structured_facts(
    {
        "id": "my-task",
        "context": Path("context.txt").read_text(encoding="utf-8"),
        "question": "在实际任务中替换为你的问题。",
    },
    client,
    config=AdaptiveV3Config(),
)
print(result["answer"])
print(client.summary())
```

在线服务需支持 OpenAI 风格 Chat Completions、完整 usage，以及精确 `/tokenize`。
普通聊天接口没有 tokenizer 时，不满足这条严格预算路径。工具 schema 与聊天模板同样参与计量；
本地检索、索引和 tokenizer 调用不计入生成 token 额度，评测时另报端到端时间。

## 输出解释

`answer` 为回答，`mode` 为执行分支，`stop_reason` 为停止依据。
结构化分支保留来源记录、`record_facts`、聚合与未知量状态；具体字段因分支而异，使用 `.get()` 查询可选字段。
`evidence_complete` 表示在已解析范围和已接纳标签成立条件下结论已确定，
`coverage_complete` 表示所需事实覆盖完整。二者都不提供独立语义真值保证。
`answer_is_estimate` 为真时，不能将外推答案当作精确计数。

正文等价复用依赖同一查询下标签只由完整投影正文决定。用户、时间或其他上下文会改变标签时，应关闭：

```python
config = AdaptiveV3Config(exact_body_coalescing=False)
```

## 本地试用

```bash
python examples/demo_harness.py
```

示例使用模拟模型与模拟 tokenizer，通过实际 V3.2 控制器验证来源绑定、复用与聚合；不进行模型能力测评。

参阅 [当前技术机制](v32_implementation.md)、[最新评测](benchmark_v32.md) 和
[生成预算实现](../long_context_agent/experiment_budget.py)。
