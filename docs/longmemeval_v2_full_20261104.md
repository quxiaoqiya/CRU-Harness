# LongMemEval-V2 Small: complete 451-question evaluation

The optional visual handoff V3 controller scores **209/451 (46.34%)**, compared
with **208/451 (46.12%)** for V1 and **72/451 (15.96%)** for original head/tail
Direct. Against V1, V3 gains four answers and loses three: a net gain of one.
The original default remains; this is an experimental visual component.

| Arm | All 451 | Previously unused 375 | Previously consumed 76 | Screenshot questions 29 | Method-assigned inference tokens |
|---|---:|---:|---:|---:|---:|
| Direct Qwen, original head/tail | 72/451 (15.96%) | 59/375 (15.73%) | 13/76 (17.11%) | 8/29 (27.59%) | 112,564,905 |
| Visual Harness V1 | 208/451 (46.12%) | 172/375 (45.87%) | 36/76 (47.37%) | 11/29 (37.93%) | 17,241,413 |
| Visual handoff V3 | **209/451 (46.34%)** | **173/375 (46.13%)** | **36/76 (47.37%)** | **12/29 (41.38%)** | **17,279,358** |

The V3–Direct difference is **30.38 percentage points**; V3 uses **84.65% fewer
method-assigned inference tokens**. This comparison does not isolate memory
retrieval from prompt/controller differences or establish universal superiority.

## Benchmark provenance and scope

- [Official LongMemEval-V2 code](https://github.com/xiaowu0162/LongMemEval-V2/tree/2cc8c540bdb87fe6761629b585e727e1c4704520), commit `2cc8c540bdb87fe6761629b585e727e1c4704520`.
- [Official dataset, pinned revision](https://huggingface.co/datasets/xiaowu0162/longmemeval-v2/tree/f152293e235517d504809563c833d7190b8c713b), revision `f152293e235517d504809563c833d7190b8c713b`.
- All **451 questions**, **Small tier: 100 history trajectories per question**. This is not the Medium tier.
- Two reused histories contain 200 trajectories and 5,095 original screenshots.
- 76 questions were previously consumed; 375 were unused before registration. Histories and interface templates are not independently held out.
- Of 29 screenshot questions, 28 had already been used. The full visual result is primarily regression evidence.
- The earlier [eight-question V3 evaluation](longmemeval_v2_visual_handoff_20261104.md) is a separate run and is part of the consumed history for this full run; its 6/8 score is not substituted into the current predictions.

All orchestration and artifacts ran on the local Mac. Only the existing Qwen
inference endpoint was called; remote inference files were not modified.

## Model and compute

All arms use `Qwen3.8-Flash-Next-FP8`, temperature 0, seed `20261103`, thinking
disabled. Direct has a **262,144-token** request/task limit, one generation and
4,096 output tokens; it retains original head/tail source spans and at most 16
original images. It is not full-history Direct. Harness has **16,384 tokens per
request, 65,536 per task, eight generations and 4,096 maximum output tokens**.
This is a same-model, different-budget comparison. Local indexing, retrieval CPU
and asset preparation costs are not converted into model tokens. Concurrent
queueing prevents interpreting recorded task durations as a controlled latency
comparison.

There are **1,353 logical method outcomes but only 931 independent execution
jobs**: 451 Direct, 451 V3/common-path and 29 independent V1 visual jobs. With no
query image, V1 and V3 call the exact same completion fallback with the same
arguments; 422 predictions and their judgments are explicitly shared. They are
not two independent stochastic replications. Method-assigned totals describe
each method's accounting on these results and must not be summed as physical
billing.

Actual physical inference: **2,093 generations / 130,747,706 tokens**. Judging is
separate: **256 generations / 330,210 tokens**. The one empty Direct response is
kept in the planned denominator without an implicit retry. Known bounded
completion recovery is metered; uncertain transport deliveries are never
silently reissued.

## Ability breakdown

| Question type | N | Direct | V1 | V3 |
|---|---:|---:|---:|---:|
| Dynamic state | 86 | 17 | 47 | 47 |
| Static state | 134 | 24 | 79 | 79 |
| Procedure | 74 | 19 | 43 | 43 |
| Errors/gotchas, screenshot-bearing | 29 | 8 | 11 | 12 |
| Dynamic state, flawed premise | 41 | 2 | 10 | 10 |
| Static state, flawed premise | 55 | 2 | 15 | 15 |
| Procedure, flawed premise | 32 | 0 | 3 | 3 |

The **128 flawed-premise questions** score **4/128** for Direct and **28/128**
for either Harness. They require identifying and explaining the incorrect
premise; a bare `UNKNOWN` is wrong under upstream rules. Generic uncertainty and
correct premise rejection are different outcomes. This remains a major weakness.

Paired against V1: 205 both correct, four V3 gains, three losses and 239 both
wrong. Against Direct: 47 both correct, 162 gains, 25 losses and 217 both wrong.
No per-question oracle selection combines methods.

## Scoring, integrity and limits

All 1,353 prediction records were fixed before scoring-label access, with macOS
file isolation during inference. Upstream deterministic matchers and LLM judge
prompts are retained; Qwen substitutes for the default LLM judge. This is not an
official leaderboard submission, LAFS score, independent human regrading or a
claim that every extra sentence is source-grounded. There are zero unresolved
judge failures in this run. Incorrect and empty answers stay in the denominator.

The independent offline audit passes **931/931 physical controller replays**,
plus the 422 explicit shared-record bindings. It verifies actual prompt/image
hashes, settings, output allowances, source fields, score-response parsing and
budgets without provider or tokenizer calls. It checks 12,191 image occurrences
and replays 26,523 tokenizer events. Mechanical replay establishes faithful
execution, not semantic entailment.

Freeze SHA-256: `05089494bc500451268c6ca47bf3b84ea453adff0c78bc60907f73c0495c5735`.
The full causal source fingerprints and budget definitions are in the public
JSON. Raw questions/answers, screenshots, provider credentials and request logs
are not included in the public artifacts.

## Artifacts and validation

[Public JSON](../benchmarks/longmemeval_v2_full_20261104.json) ·
[All 1,353 rows](../benchmarks/longmemeval_v2_full_20261104.csv) ·
[Chinese report and all 451 paired results](longmemeval_v2_full_20261104.zh-CN.md) ·
[Visual V3 API and limitations](visual_evidence_handoff_v3.md)

The public verifier requires only the repository and Python standard library:

```bash
python3 -S scripts/verify_longmemeval_v2_full_results.py
```

Full inference and recorded-I/O replay require the locally staged pinned
upstream code, data, images and experiment directory layout used by
`scripts/prepare_longmemeval_v2_full.py`; those private work artifacts are not
bundled. Provider configuration is supplied by environment variables or a local
`--provider-env` file. Credentials must not be committed.
