# LongMemEval-V2: Qwen Direct versus multimodal CRU-Harness

Completed locally on Mac: Harness **8/20 (40%)**, usable native head/tail Direct **1/20 (5%)**, a **35 percentage point** gain. Full Direct was denied locally before generation on all 20 questions because of capacity; its semantic accuracy is undefined. Only the existing Qwen inference service was called; no remote deployment files were changed.

This is a frozen **20-question small-tier pilot**: 10 questions per domain, four per ability, and four with query images. Each domain shares one 100-trajectory history: 200 unique trajectories and 5,095 history screenshots in total. These are two shared histories, not 20 independent histories, the full 451-question evaluation, or official leaderboard scores.

| Arm | Correct / planned | Returned | UNKNOWN | Mean input + output tokens | Mean end-to-end seconds |
|---|---:|---:|---:|---:|---:|
| direct_full | — (20 capacity denials) | 0 | 0 | 0 | 0.00 |
| direct_headtail_native | 1/20 (5%) | 20 | 8 | 252,478 | 23.95 |
| harness | 8/20 (40%) | 20 | 0 | 32,792 | 20.10 |

Harness consumed **655,835 input + output tokens**, versus **5,049,555** for native head/tail, an **87.01%** reduction. There were 59 and 20 generations respectively, plus 14 separately accounted judge calls and four preflight generations. No unknown usage, API failures, or budget violations occurred.

## Accuracy breakdown and failures

| Subset | Native head/tail | Harness |
|---|---:|---:|
| Deterministic official scoring | 1/12 | 5/12 |
| Qwen-judged scoring | 0/8 | 3/8 |
| Question includes an image | 0/4 | 1/4 |
| dynamic | 0/4 | 2/4 |
| gotchas | 0/4 | 1/4 |
| premise_awareness | 0/4 | 2/4 |
| procedure | 1/4 | 2/4 |
| static | 0/4 | 1/4 |
| web | 1/10 | 3/10 |
| enterprise | 0/10 | 5/10 |

Paired outcomes: one correct on both arms, seven correct only with Harness, none correct only with native head/tail, and 12 incorrect on both. Harness still has **12/20 incorrect answers and no UNKNOWNs**; native head/tail has 11 incorrect answers and eight UNKNOWNs. Abstention calibration needs improvement. Harness answered only one of four query-image questions correctly; stable visual memory ability is not established.

There were zero review parse errors and zero reported support spans omitted from final evidence. This checks mechanical retention of declared evidence, not retrieval completeness or semantic entailment. Follow-up work should separate missed retrieval, temporal state reasoning, visual grounding, and false-premise detection, then use a fresh holdout rather than relabeling tuned results as new tests.

## Source extent and budgets

- **34,235,595 Qwen canonical-text tokens**; 93,694,606 characters; 3,358 states. **130.60×** native 262,144; **2089.57×** working ceiling 16,384.
- **31,476,162 Qwen canonical-text tokens**; 77,368,369 characters; 1,737 states. **120.07×** native 262,144; **1921.15×** working ceiling 16,384.

The largest source and longest correctly answered source both contain **34,235,595 canonical-text tokens**. This is exact Qwen counting of complete trajectory JSON text, excluding screenshot pixels, and differs from paper-tokenizer units. It is neither a single model input nor a measured hard maximum. Complete text and original screenshots remain in local external memory; the model reads selected evidence windows.

Full Direct and head/tail Direct allow 262,144 tokens per request and task, one generation, and 1,024 output tokens. Head/tail retains two original text spans and uniformly samples up to 16 original screenshots from retained states; exact multimodal counting governs admission. Harness allows 16,384 tokens per request, 65,536 per task, and eight physical requests: up to three evidence reviews plus a final answer, with up to three history images per call and the query image preserved. **Compute ceilings differ**. This is not a same-budget causal estimate or an ablation against complete visual scanning and alternative retrieval.

## Opt-in adaptation and regression protection

New opt-in module `long_context_agent/trajectory_memory.py` preserves state fields, binds trajectory/state/character spans to original screenshot SHA-256 hashes, and assembles images before token admission and inference. Retrieval reuses SourceIndex and the evidence-gap protocol while accumulating validated spans. Final input uses official domain prompts and original images, without treating unverified tentative answers as facts. No generated screenshot captions, resized images, or answer caches are used.

The original harness.py, package entry point, text controllers, and structured-fact controllers are unchanged. Eleven new checks plus 89 regression checks give **100 passed**. All 58 prior LongBench source fingerprints remain unchanged. Dependencies in the original Python environment were not changed; trajectory support is imported explicitly.

## Scoring isolation, audit and verification

Metadata-stratified hash selection uses seed 20261103 without question text or gold. Preprocessing only copied labels into an isolated directory. Scoring-stage label access began after all 60 terminal inference outputs were fixed; the Mac sandbox denied the inference worker access to labels, the original answered-question file, and reports. Official scoring functions and judge prompts are pinned, but Qwen3.8-Flash-Next-FP8 replaces default gpt-5.2 as LLM judge. The eight judged questions therefore carry same-model judgment bias; no official LAFS score is claimed.

Offline audit: **60/60 ledgers**, **60/60 score re-parses**, **60/60 controller replays**, **507 image occurrences**, **79 generations** and **624 tokenizer events**. No provider/tokenizer calls during replay.

Replay uses recorded exact counts and the production controller, reparses official scores, and checks image bindings. It is not independent human semantic verification. Public JSON contains whitelisted metadata, per-call budget receipts, and hashes; raw questions, gold, model responses, images, and credentials are excluded. Public verification recomputes arithmetic/budgets and checks code/audit hashes, not unpublished answer semantics.

```bash
python3 -S scripts/verify_longmemeval_v2_results.py
```

With the complete local workspace, `python scripts/audits/audit_longmemeval_v2.py` replays offline without new generation. Inference/scoring entry: `python -m scripts.benchmark_longmemeval_v2 {freeze,run,score}`; run/score read an existing provider environment file and preserve the frozen directory. A fresh machine must obtain the pinned upstream data and prepare equivalent selection/image manifests; the report alone is insufficient to rerun inference.

## Per-question outcomes

| ID | Domain | Ability | Query image | Full Direct | Native head/tail | Harness |
|---|---|---|---|---|---|---|
| c0696888 | enterprise | dynamic | False | capacity_denied | incorrect | correct |
| 882965b7 | enterprise | dynamic | False | capacity_denied | UNKNOWN | incorrect |
| c35bfeed | enterprise | gotchas | True | capacity_denied | incorrect | incorrect |
| ce9aa351 | enterprise | gotchas | True | capacity_denied | incorrect | incorrect |
| ebad58ed | enterprise | premise_awareness | False | capacity_denied | UNKNOWN | correct |
| 19935d2d | enterprise | premise_awareness | False | capacity_denied | incorrect | correct |
| 95ae824f | enterprise | procedure | False | capacity_denied | UNKNOWN | incorrect |
| 73cbefdc | enterprise | procedure | False | capacity_denied | UNKNOWN | correct |
| f1f9de44 | enterprise | static | False | capacity_denied | UNKNOWN | incorrect |
| aef57896 | enterprise | static | False | capacity_denied | UNKNOWN | correct |
| b828a6b2 | web | dynamic | False | capacity_denied | UNKNOWN | incorrect |
| fe18a443 | web | dynamic | False | capacity_denied | incorrect | correct |
| 2cc1b9ca | web | gotchas | True | capacity_denied | incorrect | incorrect |
| 914ab1d4 | web | gotchas | True | capacity_denied | incorrect | correct |
| cdb47c5d | web | premise_awareness | False | capacity_denied | UNKNOWN | incorrect |
| a2a71197 | web | premise_awareness | False | capacity_denied | incorrect | incorrect |
| 597645be | web | procedure | False | capacity_denied | incorrect | incorrect |
| 99ae54d1 | web | procedure | False | capacity_denied | correct | correct |
| 36fe48ef | web | static | False | capacity_denied | incorrect | incorrect |
| 762d34b8 | web | static | False | capacity_denied | incorrect | incorrect |

## Benchmark provenance

[Official repository](https://github.com/xiaowu0162/LongMemEval-V2) · [Paper](https://arxiv.org/abs/2605.12493) · [Official dataset](https://huggingface.co/datasets/xiaowu0162/longmemeval-v2)

Code: `2cc8c540bdb87fe6761629b585e727e1c4704520`; data: `f152293e235517d504809563c833d7190b8c713b`; freeze SHA-256: `c9f2d218c783f4f6e71d0278c898007d3469bb34019703e77424fb14fe6cec25`.

The upstream small-file checksum manifest omits large screenshot archives; both archives were checked against pinned official LFS object hashes, then all 5,095 history images and four query images were individually verified. Upstream code is Apache-2.0; see the third-party notice. This does not establish redistribution rights for source data.

End-to-end time includes counting and generation; offline indexing is recorded separately. It is not the official memory-query latency. Qwen usage is reported without a dollar estimate. Only small tier, two shared histories, and one run are covered; results do not establish full-leaderboard or cross-model performance.

[Results JSON](../benchmarks/longmemeval_v2_qwen_20261103.json) · [Offline audit](../benchmarks/longmemeval_v2_qwen_20261103_audit.json) · [Third-party notice](../third_party/longmemeval_v2/README.md)
