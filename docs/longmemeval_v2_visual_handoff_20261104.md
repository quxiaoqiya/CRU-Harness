# Visual evidence handoff V3: measured gains and regressions

V3 scores **6/8 (75%)** on eight new screenshot questions, versus **3/8 (37.5%)**
for both direct Qwen and visual Harness V1. It retains all three correct control
answers and gains three. On 20 consumed regression questions, however, it scores
8/20 versus historical V1's 9/20. V3 remains an explicit experimental option;
the default controller is unchanged.

| New-question arm | Correct | Inference tokens | Generations |
|---|---:|---:|---:|
| Direct, original head/tail spans | 3/8 | 1,996,997 | 8 |
| Visual Harness V1 | 3/8 | 237,111 | 29 |
| Visual handoff V3 | **6/8** | **246,437** | **28** |

V3 uses 3.93% more inference tokens than V1 and 87.66% fewer than direct. These
figures exclude judging and local indexing/retrieval costs. The comparison is
same-model, different-budget: direct has a 262,144-token single-call/task limit;
both Harness arms have 16,384 tokens per call, 65,536 per task, at most eight
generations and 4,096 output tokens. Direct is truncated head/tail, not full
history. Model: `Qwen3.8-Flash-Next-FP8`, temperature 0, seed `20261103`, thinking off.

Source: [official LongMemEval-V2 code](https://github.com/xiaowu0162/LongMemEval-V2/tree/2cc8c540bdb87fe6761629b585e727e1c4704520),
[pinned dataset](https://huggingface.co/datasets/xiaowu0162/longmemeval-v2/tree/f152293e235517d504809563c833d7190b8c713b).
This is the screenshot-bearing gotchas subset, not an overall LongMemEval-V2,
LAFS or leaderboard score. New questions have no ID, normalized-text or query
image SHA overlap with the previous 68 questions. Original histories are reused;
this is question-disjoint, not source-independent evaluation. Both groups were
frozen before any new scoring access. All orchestration ran on the local Mac;
the existing Qwen endpoint was called without remote file changes.

All eight new questions use the upstream LLM gotchas checker, with Qwen replacing
its default judge model. Regression uses 19 LLM questions and one deterministic
phrase matcher. Judge outputs are reproducible, but not an independent human
regrading of answer faithfulness.

V3 repairs evidence handoff: fresh-first follow-up admission, latest-selection
priority, explicit omitted-card/image receipts, required gaps distinct from
optional searches, unresolved review state in final prompts, literal UI event
normalization, and current-versus-historical identity scope. See the
[API and implementation limits](visual_evidence_handoff_v3.md).

Remaining issues matter:

- Regression loses `af2ebaed` and `8e21c6e5`, gaining only `f2b221fd`. That gain
  satisfies a rubric accepting any alternative category; it does not establish
  the category hierarchy from evidence.
- Main known workflow failures remain wrong. Final gaps still appear in 8/20
  regression and 4/8 new V3 tasks; **none of the 28 V3 outputs explicitly abstains**.
  Reliable abstention is not demonstrated.
- Accepted answers can contain unsupported additions: alternative search syntax
  in `0574c69a`, and a personal address-book phone reinterpreted as a business
  contact number in `b339209c`. Rule correctness is not full factual grounding.
- One review is non-JSON despite normal stop; a separate final response reaches
  its 4,096-token limit. Neither is silently retried.
- Four selected regression cards and two image-critical selections remain omitted
  under fixed packet budgets, now explicitly recorded. Actual unclosed action
  tags are excluded; a complete settled multi-step workflow operator is pending.
- Several components changed together; no ablation identifies which caused gains.
  Eight new questions are too few for a broad generalization claim, and become
  consumed questions after this evaluation.

56 relevant tests pass. All **44/44 actual records** pass offline prompt/image,
source, settings, budget and score-response replay with zero provider/tokenizer
calls. Inference totals 138 generations / 3,152,319 tokens; judging separately
uses 42 generations / 50,493 tokens. Existing public integrity verifiers pass.

Artifacts: [full Chinese audit](longmemeval_v2_visual_handoff_20261104.zh-CN.md),
[public JSON](../benchmarks/longmemeval_v2_visual_handoff_20261104.json),
[all 44 rows](../benchmarks/longmemeval_v2_visual_handoff_20261104.csv).
Raw prompts, labels, screenshots and request ledgers remain local.

```bash
python3 -S scripts/verify_longmemeval_v2_visual_handoff_results.py
```
