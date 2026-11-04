# Screenshot-grounded trajectory recall

This optional controller reads the current query screenshot before searching
saved history. The [multimodal evaluation](longmemeval_v2_visual_anchor_20261103.md)
reports development and fresh-question results separately, including regressions
and the control that reads the query image without historical memory.

## Explicit use

```python
from long_context_agent.experiment_budget import ExperimentBudget
from long_context_agent.trajectory_memory_completion import PerCallBudgetedClient
from long_context_agent.visual_anchor_memory import VisualAnchorIndex, run_visual_anchor

# store: populated TrajectoryStore containing original screenshot assets
# transport: existing authenticated ProviderTransport
index = VisualAnchorIndex(store)
client = PerCallBudgetedClient(
    transport,
    ExperimentBudget(16_384, 65_536, 8, 4_096, seed=20261103),
)
result = run_visual_anchor(
    index,
    question,
    client,
    domain_system=domain_system,
    query_image="question_screenshots/current.png",
)
```

Use a new metered client for each task. `query_image` is an asset path under the
store's data root. A missing query image delegates to the existing completion
controller. Original text/fact controllers and default routes are unchanged.

The current-image pass produces bounded labels, visible state values and search
hints without a final answer. Generated perception is a retrieval hint; original
screenshots and source-bound observation fields remain in the answering request.
The current image governs current state conditions. A historical page with a
different status or role cannot establish that the current action is available.

Historical retrieval preserves state identity when accessibility text repeats
but image content differs. Consecutive before/action/after pairs retain source
state references. If the review pins one endpoint of an admitted pair, both
endpoints are retained. Physical calls, exact multimodal admission, output
reservations and actual usage share one ledger. Empty final output is recorded
without an implicit generation retry.

## Experimental effect constraints

`visual_anchor_memory_v2.py` additionally extracts original focused controls,
headings and target-state attributes, ranks exact named controls and compares
requested destination qualifiers. A comparison of the same browser ID is not a
proof that the semantic control is unchanged or an action succeeded; original
control lines and the `effect_requires_semantic_check` flag are retained.

This second version is a development experiment. It regressed from 7/12 to 6/12
on the consumed visual development cohort, so version1 was selected before the
fresh eight-question evaluation. No per-question oracle routing or combination
is used. A larger or more structured evidence packet does not itself establish
better answer accuracy.

Image hashes, source ranges, observed control relationships and recorded request
replays establish mechanical integrity. Independent human labels for exhaustive
visual relevance/recall are not supplied by this experiment. The reported scores
are for a screenshot-bearing gotchas subset of LongMemEval-V2 with a Qwen judge
substitution; they are not an overall official benchmark or LAFS score.

## 中文说明

主入口先读原始题图，提取当前可见状态和检索提示，再从历史观察字段与操作前后
状态中找证据。保留原图、来源位置和状态身份，不以生成描述替代原始证据。
没有题图时沿用已有完成控制器；默认路径不变。

第二版加入可观察的控件属性变化、聚焦控件和目标页面限定条件，但开发集出现
回归，保留为实验实现。新题在评分前已固定采用第一版，不按逐题答案取优。
本地评测与全部得失见[中文报告](longmemeval_v2_visual_anchor_20261103.zh-CN.md)。
