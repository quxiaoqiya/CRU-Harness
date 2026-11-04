# LongMemEval-V2: visual retrieval and abstention question holdout

Neither new candidate meets the replacement bar; the original default is retained. On 20 new questions, conservative routed Harness **11/20**, all-question strict gate **5/20**, previous Harness **13/20**. All six arms are retained.

Question IDs and normalized text hashes do not overlap the original 20. There are 10 questions per domain, four per ability, and four query images. The same two 100-trajectory histories are reused: 200 trajectories and 5,095 original screenshots. This is a question holdout pilot, not a source-disjoint evaluation, full 451-question run, or official LAFS/leaderboard score.

| Arm | Correct/20 | UNKNOWN | Answer coverage | Accuracy when answered | Wrong answers | Input + output tokens |
|---|---:|---:|---:|---:|---:|---:|
| direct_full | capacity denied | 0 | 0% | — | 0 | 0 |
| direct_headtail_native | 5/20 (25%) | 6 | 70% | 35.7% | 9 | 5,048,130 |
| direct_headtail_native_concise | 3/20 (15%) | 6 | 70% | 21.4% | 11 | 5,047,168 |
| harness | 13/20 (65%) | 1 | 95% | 68.4% | 6 | 610,516 |
| harness_v2 | 5/20 (25%) | 13 | 35% | 71.4% | 2 | 1,084,553 |
| harness_visual_route | 11/20 (55%) | 4 | 80% | 68.8% | 5 | 753,210 |

Full Direct capacity denials happen before generation and are not semantic wrong answers. The concise Direct control adds only a short boxed-answer requirement; source truncation, image selection, and native budget are otherwise unchanged. The experimental route does not replace the original default.

**The all-question strict gate does not improve overall accuracy and sacrifices delivery through abstention. It is not a default upgrade; the conservative route is reported separately.**

## Predeclared conservative visual route

An extension was frozen before any holdout-gold access: 16 questions without query images use the original controller; four with query images enable perceptual matching and the evidence gate. All 20 use fresh calls, without answer reuse. Bidirectional dHash and 8x8 RGB layout features are computed locally in an isolated Pillow 11.3.0 environment, without changing inference dependencies. Similarity is a heuristic, not semantic embedding or calibrated recall probability. Original PNGs are sent to inference.

No-query-image first-prompt hashes match on 16/16; raw/parsed answer hashes match on 0/16 and 11/16; new/old correct counts are 10/16 and 10/16. All 120 outputs precede any scoring-label access: True. Extension freeze: `e2252bbfa80e166e8947c275cdea45b6cafba578a312c593952dccb4624f8310`.

The no-query-image branch repeats the same old algorithm; backend outputs may vary. Differences there cannot be attributed to new visual retrieval. Runs overlap in time; timings are records, not evidence of a latency advantage.

## Scoring and image subsets

| Subset | direct_headtail_native | direct_headtail_native_concise | harness | harness_v2 | harness_visual_route |
|---|---:|---:|---:|---:|---:|
| Deterministic | 2/12 | 2/12 | 9/12 | 2/12 | 10/12 |
| Qwen judged | 3/8 | 1/8 | 4/8 | 3/8 | 1/8 |
| Query image | 3/4 | 1/4 | 3/4 | 2/4 | 1/4 |

Deterministic upstream scoring is preserved. LLM-judged items use official prompts but Qwen replaces default gpt-5.2, with possible same-model judge bias. Abstentions count as unsuccessful tasks; coverage and selective accuracy are reported separately.

All 5,099 image features were rederived and compared with cache: pass, without model calls. This does not validate semantic evidence relevance.

## Confirmed findings

On four query-image questions, the routed method corrects 1 old error but abstains on 3 old-correct questions. Strict-gate selective accuracy is 71.4%, versus 68.4% for the original, while coverage falls from 95% to 35%. This tradeoff does not establish an overall capability gain.

Known implementation limits include error-only gate development without a positive calibration cohort, quote-format rejection of recoverable candidates, and state-level seen markers blocking another field excerpt. These mechanisms need separate validation; this holdout is not retuned and reused.

## Changes and limits

1. Fuse observed-state and raw-block retrieval with trajectory-directory routing; use rarity-weighted field excerpts.
2. Bind original images to state/field/hash receipts and retrieve adjacent states for transitions.
3. Accumulate verifiable partial support, then reattach raw fields/images; disclose omitted support.
4. Distinguish supported answers, contradicted premises, and insufficient evidence; validate citations, quotes, image exposure, missing conditions, and source conflicts.
5. Render a short answer in software; allow one budgeted mechanical repair, without blind resends.

These checks combine mechanical validation and model judgments, not semantic proof. Confidence is model-reported; the 0.8 threshold is not independently calibrated. Historical agent thoughts can route raw-block retrieval but cannot serve as observed facts.

All three Harness paths share 16,384 tokens/request, 65,536/task, eight generations, and 1,024 output tokens. New/old routes allow six/three history images per call, plus the query image. Native Direct allows 262,144 tokens/request and task, one generation, and 1,024 output tokens; head/tail allows 16 images. Native/Harness compute and image exposure differ; this is not an isolated ablation.

Known limits remain: a state marked seen cannot be revisited for a different field excerpt; bounded excerpts and six final images can omit information. No per-question human oracle evidence annotations exist, so valid image bindings and image-subset accuracy do not establish improved evidence recall.

## Development history and regression protection

- longmemeval_v2_visual_dev_20261103: 0/12 correct, 11/12 UNKNOWN; 636,318 tokens; freeze `c30910f45b5ef555937073af92d75f9765e884ff81733d1b97a1f2d96789dbd3`.
- longmemeval_v2_visual_dev2_20261103: 1/12 correct, 10/12 UNKNOWN; 601,502 tokens; freeze `1731fb9c8a3e8c8ba0c6517f94341e750d7050286394c13de0fb8a3688772947`.
- longmemeval_v2_visual_dev3_20261103: 1/12 correct, 11/12 UNKNOWN; 653,464 tokens; freeze `fb7b23847dd3f8a5484cf60fc93aa65b80793c1d45ba1724c23de53e846e7ca1`.

All development stages reuse the original 12 errors and are not holdout scores. Over-abstaining failed prototypes are retained. All 236 checks pass; original text/fact controllers and first-pilot frozen sources are unchanged. No original environment dependencies, remote files, or GitHub repository were changed.

[12-case analysis](longmemeval_v2_visual_error_analysis_20261103.zh-CN.md)

Post-score deterministic diagnostic: 4 matching final candidates were rejected; 5 nonmatching candidates were rejected. This does not prove sufficient evidence for rejected candidates or permit reusing this holdout after tuning.

## Per-question outcomes

| ID | Domain | Ability | Image | direct_full | direct_headtail_native | direct_headtail_native_concise | harness | harness_v2 | harness_visual_route |
|---|---|---|---|---|---|---|---|---|---|
| 4dffe641 | enterprise | dynamic | False | capacity_denied | UNKNOWN | incorrect | incorrect | UNKNOWN | incorrect |
| f35bac4a | enterprise | dynamic | False | capacity_denied | incorrect | incorrect | correct | UNKNOWN | correct |
| af2ebaed | enterprise | gotchas | True | capacity_denied | correct | correct | correct | correct | UNKNOWN |
| 48a52262 | enterprise | gotchas | True | capacity_denied | correct | incorrect | correct | UNKNOWN | UNKNOWN |
| dbe1a814 | enterprise | premise_awareness | False | capacity_denied | UNKNOWN | UNKNOWN | incorrect | incorrect | incorrect |
| dc41b134 | enterprise | premise_awareness | False | capacity_denied | UNKNOWN | UNKNOWN | incorrect | UNKNOWN | incorrect |
| 58c34839 | enterprise | procedure | False | capacity_denied | incorrect | UNKNOWN | incorrect | UNKNOWN | correct |
| 0b50ca0d | enterprise | procedure | False | capacity_denied | incorrect | incorrect | correct | UNKNOWN | correct |
| f4b2b1fb | enterprise | static | False | capacity_denied | UNKNOWN | UNKNOWN | correct | UNKNOWN | correct |
| 9680f077 | enterprise | static | False | capacity_denied | UNKNOWN | UNKNOWN | incorrect | UNKNOWN | incorrect |
| 96497069 | web | dynamic | False | capacity_denied | correct | correct | correct | UNKNOWN | correct |
| 5c826b28 | web | dynamic | False | capacity_denied | UNKNOWN | UNKNOWN | correct | UNKNOWN | correct |
| 910135bb | web | gotchas | True | capacity_denied | correct | incorrect | correct | UNKNOWN | UNKNOWN |
| 07a0145f | web | gotchas | True | capacity_denied | incorrect | incorrect | incorrect | correct | correct |
| e532c01e | web | premise_awareness | False | capacity_denied | incorrect | incorrect | UNKNOWN | UNKNOWN | UNKNOWN |
| aa8d21f5 | web | premise_awareness | False | capacity_denied | incorrect | incorrect | correct | correct | incorrect |
| 8957a127 | web | procedure | False | capacity_denied | incorrect | incorrect | correct | incorrect | correct |
| 92642fe7 | web | procedure | False | capacity_denied | correct | correct | correct | UNKNOWN | correct |
| 0738c1ac | web | static | False | capacity_denied | incorrect | incorrect | correct | correct | correct |
| b8d56b89 | web | static | False | capacity_denied | incorrect | incorrect | correct | correct | correct |

## Provenance, audit and verification

[Official repository](https://github.com/xiaowu0162/LongMemEval-V2) · [Paper](https://arxiv.org/abs/2605.12493) · [Official data](https://huggingface.co/datasets/xiaowu0162/longmemeval-v2)

Data `f152293e235517d504809563c833d7190b8c713b`; code `2cc8c540bdb87fe6761629b585e727e1c4704520`; freeze `8a9b321daee247c0429b5a4065f76b7bc70f040a970d033844aa1836073a5811`.

Offline audit: 120/120 ledgers, 120/120 score re-parses, 120/120 controller replays; status pass. Replay uses saved counts/responses and frozen production logic, without provider calls or independent human semantic validation.

```bash
python3 -S scripts/verify_longmemeval_v2_visual_results.py
```

Largest source remains 34,235,595 Qwen canonical-JSON text tokens, excluding pixels; this run does not expand source-history size. Timing includes tokenization and inference, with indexing separate, not official query latency. Public artifacts omit raw questions, gold, images, replies, and credentials; full local replay requires the workspace.

[Results JSON](../benchmarks/longmemeval_v2_visual_holdout_20261103.json) · [Primary audit](../benchmarks/longmemeval_v2_visual_holdout_20261103_audit.json) · [Route audit](../benchmarks/longmemeval_v2_visual_holdout_20261103_route_audit.json)
