# LongMemEval-V2 visual evaluation: current-state perception and action outcomes

This evaluation covers screenshot-bearing gotchas questions only. Twelve previously consumed visual questions form development; eight new visual questions exclude the prior 60 IDs/normalized texts. The same two 100-trajectory histories are reused; this is not source- or template-disjoint.

## New visual-question results

| Arm | Correct | UNKNOWN | Inference tokens | Generations |
|---|---:|---:|---:|---:|
| direct_headtail_native | 2/8 (25.0%) | 0 | 1,996,847 | 8 |
| harness_completion | 1/8 (12.5%) | 1 | 267,907 | 24 |
| visual_query_only | 0/8 (0.0%) | 1 | 24,465 | 16 |
| harness_visual_anchor | 2/8 (25.0%) | 0 | 197,643 | 26 |

The candidate retains 1/1 baseline-correct questions, gains 1, loses 0, and uses 26.23% fewer inference tokens. It still misses 6/8; these results do not establish general visual superiority or reliable abstention.

Exact query-image hash overlap with development: 0. This does not exclude shared histories, UI templates or perceptually similar images.


Inference usage excludes separately metered scoring. Three 16k arms share 16,384/request, 65,536/task, eight generations and 4,096 maximum output; actual usage differs. Head/tail Direct has larger 262,144 request/task ceilings, one generation and up to 16 original images.

All four arms score 0/4 in the enterprise domain; Direct and the candidate score 2/4 on web questions. Successful perception-JSON parsing does not establish accurate perception or separate every retrieval failure from reasoning failure.

## Implementation and controls

The holdout candidate is fixed to visual-recall version 1 before any new result: development 7/12 versus version 2 at 6/12. Version 2 fixes the rating-widget case but regresses on report-module and filter questions, so it is not selected. Its code, scores and full audit are retained.

An original-query-image pass extracts visible labels, state values and retrieval hints without answering. Generated notes route retrieval; final inference still receives the original image. Historical retrieval uses observed fields and before-control/action/after-state pairs. Identical text with different images retains distinct states; pinning a transition retains both endpoints. Final inference uses native reasoning without a confidence gate.

visual_query_only shares query perception and final policy but supplies no history, testing whether improvements come only from closer image reading. Per-question comparison prevents attributing all gains to memory. Current-image values govern current preconditions; similar historical pages cannot override them.

## Paired outcomes

| Control | Both correct | Anchor only | Control only | Neither |
|---|---:|---:|---:|---:|
| harness_completion | 1 | 1 | 0 | 6 |
| visual_query_only | 0 | 2 | 0 | 6 |

## Development regression

### Version1: selected for holdout

| Arm | Correct/12 | UNKNOWN | Inference tokens |
|---|---:|---:|---:|
| harness_completion | 6/12 | 0 | 350,590 |
| visual_query_only | 4/12 | 1 | 42,510 |
| harness_visual_anchor | 7/12 | 0 | 351,982 |

### Version 2: not selected after regressions

| Arm | Correct/12 | UNKNOWN | Inference tokens |
|---|---:|---:|---:|
| harness_completion | 6/12 | 1 | 320,292 |
| visual_query_only | 2/12 | 0 | 44,236 |
| harness_visual_anchor | 6/12 | 0 | 431,564 |

The first development version scores completion 6/12, query-only 4/12, visual recall 7/12; records are preserved. The second version adds observed target-attribute effects, focused controls and destination qualifiers.


Development is used for tuning, not independent generalization. Original defaults and frozen records remain unchanged; this module is opt-in. Evaluation is orchestrated on the Mac without changing inference-server files; code and sanitized results are published separately after evaluation.

## Per-question outcomes and diagnostics

| ID | Domain | Direct | Completion | Query only | Visual anchor |
|---|---|---|---|---|---|
| 7586cf7c | enterprise | incorrect | incorrect | incorrect | incorrect |
| 285db8ec | enterprise | incorrect | incorrect | incorrect | incorrect |
| 50d92f55 | enterprise | incorrect | incorrect | incorrect | incorrect |
| 18b91103 | enterprise | incorrect | incorrect | incorrect | incorrect |
| f2b221fd | web | correct | incorrect | incorrect | incorrect |
| 81180ae9 | web | incorrect | incorrect | incorrect | incorrect |
| 626e401e | web | correct | UNKNOWN | UNKNOWN | correct |
| b339209c | web | incorrect | correct | incorrect | correct |

harness_visual_anchor: transition cases 2/8; perception parse errors 0.

visual_query_only: transition cases 0/8; perception parse errors 0.

Data documentation and recorded action alignment differ. Of 594 fill actions checked, 531 retain the control on both sides; 495 show the literal value only after, and 0 show the reverse. The adapter pairs the action at state i with i-1→i. This diagnostic does not prove every action succeeds.

## Audit and scope

All 56 targeted regression checks pass. Offline replays pass 36/36 second development, 36/36 preliminary development and 32/32 holdout records, checking original image bytes, source fields and paired states. Each stage fixes all outcomes before its scoring-label access. Mechanical integrity is not human semantic relevance/recall annotation.

Qwen3.8-Flash-Next-FP8, temperature=0, seed=20261103, thinking disabled. Official data/scoring prompts are pinned; these questions use Qwen instead of default gpt-5.2 as LLM judge. Eight new questions, two reused histories and one run do not establish general/significant superiority. This is not an overall LongMemEval-V2, official LAFS or leaderboard score.

Preliminary development scoring stopped on missing question_type report metadata. A separate adapter only supplies a gotchas default for that field, preserving inference records, reference answers, official judge prompts and completed judge calls. Holdout metadata was repaired before its freeze.

[Official code](https://github.com/xiaowu0162/LongMemEval-V2) · [Dataset](https://huggingface.co/datasets/xiaowu0162/longmemeval-v2)

Data `f152293e235517d504809563c833d7190b8c713b`; upstream `2cc8c540bdb87fe6761629b585e727e1c4704520`.

```bash
python3 -S scripts/verify_longmemeval_v2_visual_anchor_results.py
```

[Results JSON](../benchmarks/longmemeval_v2_visual_anchor_20261103.json) · [Version1 development audit](../benchmarks/longmemeval_v2_visual_anchor_20261103_preliminary_development_audit.json) · [Version 2 development audit](../benchmarks/longmemeval_v2_visual_anchor_20261103_development_audit.json) · [Holdout audit](../benchmarks/longmemeval_v2_visual_anchor_20261103_holdout_audit.json) · [Opt-in API](visual_anchor_memory.md)
