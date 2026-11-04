# Trajectory completion and reversible evidence display

These are explicit experimental controllers for the existing multimodal trajectory
store. They preserve the original trajectory controller and its default route.
The [frozen LongMemEval-V2 report](longmemeval_v2_compact_holdout_20261103.md)
contains results, budgets, failed development trials and per-question outcomes.

## Completion allocation

`CompletionEvidenceClient` reuses the original retrieval and review loop. At the
final request it tries 4,096, 2,048 and 1,024 output tokens, keeping the complete
baseline evidence and original images. It does not discard evidence to enlarge
the decoder reservation. `PerCallBudgetedClient` ensures that the resource notice
in both tokenization and generation advertises the actual request's allowance.

A valid, non-UNKNOWN boxed native answer is preserved. A short unboxed answer
may be wrapped without generation. Otherwise one 512-token recovery is allowed
inside the original task budget. Invalid or unaffordable recovery retains the
original candidate. UNKNOWN recovery additionally requires a newly admitted
state image and an exact short quote present in an attached state's observed
accessibility text. Quote presence is a mechanical check, not semantic entailment.
The primary request keeps native reasoning order; only exception recovery asks
for the answer first.

## Reversible display

`CompactCompletionClient` adds a final-prompt presentation step. JSON newline,
tab and quoted-text escapes are displayed literally. Overlapping source ranges
may share their coordinate union, with original headers retained as aliases.
Identical strings at distinct source positions are retained. The external source
is unchanged and every transformed range round-trips to its original characters.

Reviews remain byte-identical to the original implementation. The final prompt
keeps the baseline source-position union and original images. Exact token counts
include the display notice, multimodal payload and actual output notice. A
source-header mismatch or lack of token savings falls back to raw presentation.
Additional original images are permitted only during exception recovery.

Literal escape display is not valid-JSON serialization, a generated summary, a
larger attention window, or proof of better retrieval. Source-position integrity
does not establish that the retrieved ranges contain the necessary answer.

## Explicit opt-in

Use a populated `TrajectoryStore` whose root contains its original screenshot
assets. Only trajectories enter `store.insert`; benchmark questions, reference
answers and scoring functions must stay outside the store. Keep dataset labels
out of the inference process.

```python
from long_context_agent.experiment_budget import ExperimentBudget
from long_context_agent.trajectory_memory_completion import (
    PerCallBudgetedClient,
    CompletionEvidenceClient,
    run_completion_harness,
)
from long_context_agent.trajectory_memory_compact import (
    CompactCompletionClient,
    run_compact_completion,
)

# transport: existing authenticated ProviderTransport
# store: populated TrajectoryStore; question: user question
budget = ExperimentBudget(
    context_tokens=16_384,
    total_tokens=65_536,
    max_calls=8,
    max_output_tokens=4_096,
    seed=20261103,
)
metered = PerCallBudgetedClient(transport, budget)
adapter = CompactCompletionClient(
    metered,
    store,
    domain_system=domain_system,
    question_text=question,
    query_image=query_image,  # relative asset path or None
)
result = run_compact_completion(store, question, adapter)
```

For the completion-only ablation, replace the final two constructor/function
names with `CompletionEvidenceClient` and `run_completion_harness`. Use a fresh
metered client for each task/method. Do not reuse a consumed budget or select the
best answer across experimental routes after seeing reference labels.

The dedicated benchmark runner is
[`scripts/benchmark_longmemeval_v2_compact.py`](../scripts/benchmark_longmemeval_v2_compact.py).
It uses a frozen selection, macOS label-access isolation, physical-request
journals, one-shot generation and scoring only after all registered terminals.
Inference logs, labels, source histories, screenshots and credentials stay local.
Public verification performs no network calls:

```bash
python3 -S scripts/verify_longmemeval_v2_compact_results.py
```

## 中文说明

新增控制器为显式实验入口，原默认路径不变。完成版在不删除原证据／原图的
前提下分配最终输出额度，原生有效答案不接受二次裁判覆盖；恢复失败保留原答。
可逆展示版只改变最终证据的显示方式，保留来源位置与不同观察的数量，严格核对
真实多模态提示计数，不以摘要替换原文。两者都共享原任务总预算。

引用存在、可逆还原与图片哈希属于机械完整性；它们不等于语义支持或视觉召回。
完整实验的真实准确率、错误回归、输出截断、UNKNOWN 和用量见
[中文报告](longmemeval_v2_compact_holdout_20261103.zh-CN.md)。
