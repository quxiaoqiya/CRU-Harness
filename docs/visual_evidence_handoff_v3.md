# Visual evidence handoff V3 / 视觉证据交接 V3

V3 is an explicit, opt-in candidate. It addresses the evidence-flow defects in
the [forensic audit](longmemeval_v2_visual_forensic_audit_20261104.zh-CN.md).
The original text controller, V1, V2 and their frozen experiments are unchanged.
Mechanical integrity and answer accuracy are evaluated separately.

The [completed evaluation](longmemeval_v2_visual_handoff_20261104.zh-CN.md)
reports 6/8 on new questions versus 3/8 for direct/V1, but 8/20 versus the
historical V1's 9/20 on regression questions. V3 remains opt-in.

```python
from long_context_agent.visual_anchor_memory_v3 import (
    AnchorConfig, VisualAnchorIndex, run_visual_anchor,
)
from long_context_agent.trajectory_memory_completion import PerCallBudgetedClient
from long_context_agent.experiment_budget import ExperimentBudget

index = VisualAnchorIndex(store, AnchorConfig())
client = PerCallBudgetedClient(
    transport, ExperimentBudget(16_384, 65_536, 8, 4_096, seed=20261103),
)
result = run_visual_anchor(
    index, question, client,
    domain_system=domain_system,
    query_image="question_screenshots/current.png",
)
```

Use an independent metered client per task. Source assets remain under the
`TrajectoryStore` data root. Without a question image this entry point delegates
to the original completion route.

## What changed

| Defect | V3 behavior | Remaining boundary |
|---|---|---|
| Old pins occupy the final packet before a follow-up is admitted | Fresh cards lead the next review; latest valid selections lead final admission | Six text cards cannot retain every selection; omitted selected IDs are reported |
| `complete=true` hides missing facts | Required gaps, optional searches and unresolved conflicts are distinct; gaps/conflicts invalidate completion | These are model-produced hypotheses, not a proof of entailment |
| Review concerns disappear at the answer boundary | The final request includes the latest review, missing facts, conflicts and omitted IDs | The final model must still interpret the observations correctly |
| UI dialects and thought strings mix | Literal-only parsing supports `press`, `clear`, focus, tab changes and ordered calls; raw narration is not rendered as an observed action | Malformed/unclosed action tags are excluded; no source code is executed |
| Static similarity hides an independent specific search | Each independent query keeps a leading unseen hit before rank fusion; visible state values help route searches | A routing value is not evidence; retrieval can still select the wrong branch |
| Current object is conflated with a historical object | Each history card is scoped as a historical example; final instructions require identity/precondition checks | This is a scope constraint, not a deterministic entity resolver |
| Text-card retention is mistaken for pixel exposure | Reviews nominate image-critical IDs; SHA-equivalent aliases are recorded; the final packet lists text-only cards | Three history-image slots cannot expose all selected images |

There is no confidence threshold, answer-first JSON gate, implicit generation
retry, answer-key routing or replacement of original screenshots with generated
captions. Exact admission reserves output tokens before each physical request.

## Receipts

- `trace[].fresh_ids`, `selected_ids`, `priority_ids`: follow-up and selection flow.
- `latest_review`: required gaps, conflicts, coverage hypotheses and searches.
- `pinned_omitted`: **latest selected** evidence missing from the actual final packet.
- `final_visual_omitted`: image-critical selections whose original pixels were absent.
- `image_events`: original image SHA, aliases, source-field ranges and actual prompt hashes.

The word `pinned` is retained for compatibility with earlier result shapes. In
V3 it tracks latest selections; it does not claim every historical pin survives.

## 中文说明

这次修复的核心是把“追检到了什么、最后采用了什么、还有什么没解决”可靠地交给
最终回答。新卡不再只能排在旧卡后面；审核的必要缺口、冲突和预算遗漏不会在最后
一轮被静默丢弃。历史实体明确作为类比例子，不能直接证明当前用户操作过另一条记录。

对 5095 个历史状态的独立检查中，非 `noop` UI 动作可解析数从 3610 增至 4074，新增
464 条；包含 260 条 `noop` 时合计 4334。`tab_focus` 指浏览器标签页切换，不能把它
直接等同于表单字段失焦。动作可解析只说明来源可核对，不代表执行成功。

本版尚未实现完整的多步工作流和稳定结果链算子，也没有恢复实际数据里 26 条未闭合
的动作标签。对这些记录采用安全排除，避免把代理思考混入观察证据。不要把本版描述为
已经解决全部视觉召回问题。

实验使用固定版 LongMemEval-V2 的带截图 gotchas 子集，先做 20 题诊断回归，再做
8 道未参与此前调优的新题三路对照。新题与此前 68 题的题号、规范化题文、题图 SHA
无重叠，但复用历史轨迹，因此不属于来源独立测试。完整结果由本地冻结实验及独立
零 API 重放审计生成，不能由单元测试通过推断准确率提高。
