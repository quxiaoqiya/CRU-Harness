# LongMemEval-V2: native reasoning budget and reversible evidence display

New 20 questions: head/tail Direct 4/20, original Harness 8/20, native completion 11/20, reversible display 11/20.

Zero question-ID/normalized-text overlap with the prior 40. Four questions per ability, ten per domain, four query images. Two 100-trajectory histories are reused: 200 trajectories and 5,095 original history images, not a source-disjoint holdout. All 100 outcomes preceded scoring-label access.

| Arm | Correct/20 | UNKNOWN | Coverage | Selective accuracy | Input + output tokens |
|---|---:|---:|---:|---:|---:|
| direct_full | capacity denied | 0 | 0% | — | 0 |
| direct_headtail_native | 4/20 (20%) | 8 | 60% | 33.3% | 5,002,380 |
| harness | 8/20 (40%) | 1 | 95% | 42.1% | 650,438 |
| harness_completion | 11/20 (55%) | 2 | 90% | 61.1% | 689,829 |
| harness_compact | 11/20 (55%) | 2 | 90% | 61.1% | 685,870 |

Full Direct capacity denials are not semantic wrong answers. Direct allows 4,096 output tokens, 262,144/request and task, one generation. Harness ceilings are shared: 16,384/request, 65,536/task, eight generations, 4,096 maximum output. The original controller requests 1,024; completion allocates 4,096/2,048/1,024 only when admissible without dropping evidence. Output allocation is part of the treatment; actual usage is separate.

The two complete canonical JSON history texts contain 31,476,162 and 34,235,595 Qwen tokens, excluding image pixels. Harness retrieves from complete external sources; each model call receives bounded evidence. These source lengths are neither single attention windows nor accurate-context expansion factors. All arms share per-call notices of the actual output reservation.

Model: Qwen3.8-Flash-Next-FP8; temperature=0, generation seed=20261103, enable_thinking=false, preserve_thinking=false.

Usage tables cover inference input plus output, excluding scoring. Separate judge ledgers record 26 generations and 38,827 tokens, outside method task budgets. Tokenizer preflight and offline replay are not generations.

## Findings and remaining failures

The two candidates retain 8/8 and 8/8 baseline-correct questions, adding 3 and 3 correct answers. The completion-adaptation bundle improves observed accuracy by 15 percentage points; this pilot cannot isolate one causal mechanism.

Completion and display both score 11/20, with 4 correctness disagreements and no per-question oracle combination. Display saves admission space without an additional observed accuracy gain. Both candidates score only 1/4 on query-image questions. No UNKNOWN answer was rescued through a new image plus quote; semantic abstention calibration remains unresolved.

## Changes and invariants

Preserve native reasoning order, without answer-first forcing or confidence gates. Original retrieval/reviews are reused. Resource notices reflect each actual output request. A valid non-UNKNOWN native box is never overridden by a second judge. One recovery is permitted for missing format; UNKNOWN rescue requires new admitted state images and a short quote present in the corresponding observed field. Failed recovery retains the original candidate.

Final evidence only: render JSON newline/tab/quote escapes and merge coordinate-overlapping ranges with original-header aliases. Identical text at distinct positions is retained, preserving observation identity/multiplicity. Display maps restore original source characters; original images are unchanged. This is not generative summarization or a larger attention window.

Quote checks establish presence in an observed field, not answer entailment. UNKNOWN counts and selective accuracy are reported. There are no independent human visual-relevance labels, so image integrity is not a visual semantic recall improvement.

Matched complete prompts with display transforms total 293,256 → 275,244 tokens: 18,012 saved (6.14%). On 9 requests the raw representation would not fit with the same output reservation. Duplicate coordinate characters removed: 5,280; zero means no measured overlap-dedup benefit.

This ratio covers matched complete prompts that were transformed, not end-to-end total-token reduction. Retrieval rounds, evidence paths and actual outputs can differ; the table above reports end-to-end input plus output usage.

Complete-prompt admission counts each include a 16-token safety margin, which cancels in paired differences. Source-text preflight uses separate raw tokenizer counts.

The default trajectory controller remains unchanged. Completion/display controllers are explicit experimental entry points, without routing learned from these 20 questions. One run per question/method; no per-question best-of selection, voting, or post-test answer edits.

| Candidate vs original | Both correct | Candidate only | Original only | Neither |
|---|---:|---:|---:|---:|
| harness_completion | 8 | 3 | 0 | 9 |
| harness_compact | 8 | 3 | 0 | 9 |

| Candidate | Native valid boxes overridden | Recovery calls | Supplemental image cases | Final output reservations |
|---|---:|---:|---:|---|
| harness_completion | 0 | 4 | 2 | {"1024": 10, "2048": 2, "4096": 8} |
| harness_compact | 0 | 2 | 1 | {"1024": 10, "2048": 3, "4096": 7} |

| Arm | Length-truncated generations | Unboxed final answers | Recovery outcomes |
|---|---:|---:|---|
| harness | 6 | 6 | {} |
| harness_completion | 6 | 3 | {"budget_unavailable_kept_native": 1, "format_recovered": 2, "invalid_recovery_kept_native": 2, "native_answer_kept": 13, "refusal_kept_no_new_observation": 2} |
| harness_compact | 4 | 2 | {"budget_unavailable_kept_native": 1, "format_recovered": 1, "invalid_recovery_kept_native": 1, "native_answer_kept": 15, "refusal_kept_no_new_observation": 2} |

## Development and positive retention

| Candidate | Correct/20 | Old correct retained/8 | Old errors recovered/12 |
|---|---:|---:|---:|
| harness_completion | 9 | 7 | 2 |
| harness_compact | 8 | 7 | 1 |
| failed harness_format | 7 | 5 | — |
| failed harness_assist | 8 | 4 | — |

Development reuses the original 20 (eight correct, twelve wrong), not holdout scores. Failed answer-first trials are retained. A separate source-text-only preflight measures 10,383→9,357 tokens (9.88%), not a complete multimodal-prompt ratio.

## Subsets and per-question outcomes

| Subset | Direct | Original | Completion | Compact |
|---|---:|---:|---:|---:|
| Deterministic | 3/12 | 6/12 | 8/12 | 8/12 |
| Qwen judged | 1/8 | 2/8 | 3/8 | 3/8 |
| Query image | 1/4 | 0/4 | 1/4 | 1/4 |
| ability: dynamic | 0/4 | 1/4 | 2/4 | 2/4 |
| ability: gotchas | 1/4 | 0/4 | 1/4 | 1/4 |
| ability: premise_awareness | 0/4 | 2/4 | 2/4 | 2/4 |
| ability: procedure | 0/4 | 3/4 | 3/4 | 4/4 |
| ability: static | 3/4 | 2/4 | 3/4 | 2/4 |
| domain: enterprise | 2/10 | 4/10 | 6/10 | 5/10 |
| domain: web | 2/10 | 4/10 | 5/10 | 6/10 |

| ID | Domain | Ability | Image | Direct | Original | Completion | Compact |
|---|---|---|---|---|---|---|---|
| d0d61088 | enterprise | dynamic | False | UNKNOWN | incorrect | incorrect | incorrect |
| 5edd2533 | enterprise | dynamic | False | incorrect | correct | correct | correct |
| 8e21c6e5 | enterprise | gotchas | True | correct | incorrect | correct | correct |
| 2f6a21cd | enterprise | gotchas | True | incorrect | incorrect | incorrect | incorrect |
| 5849c18b | enterprise | premise_awareness | False | UNKNOWN | correct | correct | correct |
| 571b841c | enterprise | premise_awareness | False | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN |
| 025db8ef | enterprise | procedure | False | UNKNOWN | correct | correct | correct |
| 25b00876 | enterprise | procedure | False | UNKNOWN | correct | correct | correct |
| a444cec8 | enterprise | static | False | correct | incorrect | incorrect | UNKNOWN |
| 2975030d | enterprise | static | False | UNKNOWN | incorrect | correct | incorrect |
| 5a85ca0f | web | dynamic | False | incorrect | incorrect | correct | incorrect |
| 11dac74b | web | dynamic | False | UNKNOWN | incorrect | UNKNOWN | correct |
| 1e77eb4b | web | gotchas | True | incorrect | incorrect | incorrect | incorrect |
| 1d864942 | web | gotchas | True | incorrect | incorrect | incorrect | incorrect |
| 1a89c3ac | web | premise_awareness | False | incorrect | incorrect | incorrect | incorrect |
| 922a0c56 | web | premise_awareness | False | UNKNOWN | correct | correct | correct |
| b4dcf650 | web | procedure | False | incorrect | correct | correct | correct |
| 99fa7c96 | web | procedure | False | incorrect | incorrect | incorrect | correct |
| 7c8efee5 | web | static | False | correct | correct | correct | correct |
| 26a038ce | web | static | False | correct | correct | correct | correct |

## Audit and scope

258 checks passed. Offline audit: 100/100 ledgers, 100/100 score re-parses, 100/100 controller replays; status pass. Original images, source-position union, round trips and actual output/admission are checked, not independent human entailment.

One run on 20 questions/two shared histories does not establish general significant superiority. Fixed seed/temperature zero may still vary. Harness hard ceilings match; actual input/output allocations differ, and Direct has a larger budget. Original controllers/results, dependencies and remote files remain unchanged. Pinned official deterministic scoring/prompts are used, with Qwen replacing default gpt-5.2 as judge; not official LAFS or an online leaderboard score.

[Official code](https://github.com/xiaowu0162/LongMemEval-V2) · [Paper](https://arxiv.org/abs/2605.12493) · [Dataset](https://huggingface.co/datasets/xiaowu0162/longmemeval-v2)

Data `f152293e235517d504809563c833d7190b8c713b`; code `2cc8c540bdb87fe6761629b585e727e1c4704520`; freeze `cef31f1ac7da7b732f229d44ca22211188dda9703d9360cb0d34c2b543095ac7`.

```bash
python3 -S scripts/verify_longmemeval_v2_compact_results.py
```

[Results JSON](../benchmarks/longmemeval_v2_compact_holdout_20261103.json) · [Offline audit](../benchmarks/longmemeval_v2_compact_holdout_20261103_audit.json) · [Explicit opt-in API](trajectory_completion_experiments.md)
