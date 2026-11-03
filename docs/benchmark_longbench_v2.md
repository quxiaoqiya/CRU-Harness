# LongBench v2: historical evaluation and capacity audit

**English** | [简体中文](benchmark_longbench_v2.zh-CN.md)

CRU-Harness's historical evidence-window workflow answered **313/503 questions correctly (62.23%)**, versus **254/503 (50.50%)** for direct Qwen: **59 additional correct answers and an 11.73 percentage-point increase in end-to-end accuracy**. In the latest dedicated rerun of the historical 82-question long-context selection, it answered **50/82 correctly (60.98%)**, versus **1/82 (1.22%)** for the cached direct baseline.

The clearest capacity result is inside that 82-question rerun: direct Qwen rejected **79 prompts** with an explicit context-limit error, while Harness answered **49 of those 79 correctly (62.03%)**. This demonstrates a working answer path for tasks blocked by the direct service's input limit. The gain combines recovery from capacity failures with retrieval and multiple inference calls; it does not isolate a gain in the base model's reasoning ability at equal compute.

These are **locally executed evaluations with real Qwen inference**, published from retained experiment artifacts. They are **not an official LongBench leaderboard submission**. They evaluate the earlier evidence-window Harness workflow, not a new LongBench rerun of the current V3.2 structured-fact and semantic-aggregation path. The V3.2 OOLONG results are documented separately in [the V3.2 report](benchmark_v32.md).

## Benchmark source and publication record

LongBench v2 is an external benchmark created by Bai et al., with 503 multiple-choice questions across six domains. Its dataset length categories are expressed in **words**, not Qwen tokens. Sources: [paper](https://arxiv.org/abs/2412.15204), [official project and leaderboard](https://longbench2.github.io/), [official code](https://github.com/THUDM/LongBench), and [official dataset](https://huggingface.co/datasets/zai-org/LongBench-v2). The Hugging Face split is named `train`; this historical evaluation uses all 503 rows of that public benchmark split.

The [public results JSON](../benchmarks/longbench_v2_historical.json) provides per-question predictions, expected choices, source context hashes, aggregate results, and hashes of the retained local artifacts. It publishes derived results without source contexts, complete questions, model traces, credentials, or private machine paths. The dataset revision and pre-run code commit were not recorded in the historical artifacts; present-day artifact hashes cannot retrospectively establish a pre-run freeze. We do not invent a pinned revision for this experiment.

Two runs remain separate:

- **Full historical run:** all 503 questions, from `longbench_v2_full`.
- **Latest dedicated long-context rerun:** the previously selected 82 questions, from `longbench_v2_over260k_uniform_v2`; the retained final audit has no targeted per-question overrides. Its Direct records reuse the earlier cache.

The 82 rerun results are not spliced into the 503 run to present a synthetic full-dataset score. This subset was examined and repeatedly used for optimization, so it is a regression set rather than an unseen holdout.

## Aggregate results

“All-task accuracy” is correct answers divided by every attempted question. Missing answers and rejected requests stay in its denominator. “Answered-only accuracy” uses only rows with a parsed choice.

| Run | Method | Parsed answers | Correct | All-task accuracy | Answered-only accuracy |
|---|---|---:|---:|---:|---:|
| Full 503 | Direct Qwen | 401/503 | 254 | **50.50%** | 63.34% |
| Full 503 | Historical Harness | 502/503 | 313 | **62.23%** | 62.35% |
| Long-context 82 rerun | Cached Direct Qwen | 3/82 | 1 | **1.22%** | 33.33% |
| Long-context 82 rerun | Historical Harness, improved retrieval and verification | 81/82 | 50 | **60.98%** | 61.73% |

All 102 direct failures in the 503 run and all 79 direct failures in the 82 rerun contain an explicit context-limit error. These are delivery failures of the direct full-prompt path, not 102 or 79 observed incorrect model answers. Harness answered 59/102 of the rejected full-run questions correctly and 49/79 of the rejected subset questions correctly.

The full run reused cached Direct results for **394 `direct_pass`** rows; its other paths were **98 `raw_source`** and **11 `evidence_fallback`**. The 82 rerun used **81 `raw_source`** rows and **1 cached `direct_pass`** row. Thus, many Direct/Harness entries share the same generation. The methods are paired execution paths, not fully independent repetitions.

The 82 rerun increased answer coverage by **78 questions** and correct answers by **49 questions** versus its cached Direct baseline. Option verification covered **81 questions and 242 views**; there was **1 parsing retry** and **1 residual parsing/request error**. No option-verification override was recorded. An individual failed verification view is distinct from a missing final answer; one Harness question still had no parsed choice.

## Results by domain: full 503-question run

| Domain | Questions | Direct correct / all-task accuracy | Harness correct / all-task accuracy |
|---|---:|---:|---:|
| Code Repository Understanding | 50 | 13/50 · 26.00% | 32/50 · 64.00% |
| Long In-context Learning | 81 | 39/81 · 48.15% | 55/81 · 67.90% |
| Long Structured Data Understanding | 33 | 12/33 · 36.36% | 19/33 · 57.58% |
| Long-dialogue History Understanding | 39 | 30/39 · 76.92% | 30/39 · 76.92% |
| Multi-Document QA | 125 | 63/125 · 50.40% | 69/125 · 55.20% |
| Single-Document QA | 175 | 97/175 · 55.43% | 108/175 · 61.71% |

## Results by domain: 82-question rerun

| Domain | Questions | Direct correct / all-task accuracy | Harness correct / all-task accuracy |
|---|---:|---:|---:|
| Code Repository Understanding | 31 | 0/31 · 0.00% | 19/31 · 61.29% |
| Long In-context Learning | 12 | 0/12 · 0.00% | 9/12 · 75.00% |
| Long Structured Data Understanding | 9 | 0/9 · 0.00% | 5/9 · 55.56% |
| Multi-Document QA | 14 | 0/14 · 0.00% | 6/14 · 42.86% |
| Single-Document QA | 16 | 1/16 · 6.25% | 11/16 · 68.75% |

The 82 rerun's Multi-Document QA result is **6/14 (42.86%)**. Improved document coverage and verifier parsing did not establish an accuracy increase in that domain. Code repository questions reached **19/31 (61.29%)** and long in-context learning reached **9/12 (75.00%)**. These are subset measurements with different difficulty and length composition from the full dataset.

## Length measurement and the “260k+” selection

The historical selection rule was `estimated_prompt_tokens > 260000 OR cached_context_tokens > 260000`. The cached field mixed two sources:

- **79/82 rows:** `char_estimate`, based on source character count divided by four.
- **3/82 rows:** `provider_prompt_tokens`, from cached Direct usage. For indices **64 and 307**, the helper combined usage from the primary and output-recovery calls; that value is not a single request's prompt length.

The older audit called its copied cached field `provider_prompt_tokens` even when the value came from a character estimate. The public report preserves the actual source distinction. The 82 rows are therefore a **historical 260k+ selection under mixed measurements**, not 82 independently tokenizer-verified inputs above the 262,144-token routing limit. The selection threshold (260,000) and routing limit (262,144) also differ.

The largest retained source has **16,182,936 characters**. Its historical **4,045,734-token** value is a **characters/4 estimate**, not a measured Qwen token count; it cannot establish a measured native-context expansion multiple. Explicit context-limit rejections provide stronger evidence of the direct path's capacity boundary here. The accurately tokenized Qwen expansion measurements reported for V3.2 OOLONG belong to a separate experiment.

## Retained execution configuration and scoring

| Setting | Full 503 run | Dedicated 82 rerun |
|---|---|---|
| Model | `Qwen3.8-Flash-Next-FP8` | Same named model |
| Harness routing context / output reserve | 262,144 / 8,192 tokens | Same |
| Answer maximum / candidate maximum | 8,192 / 2,048 tokens | Same |
| Source chunk size | 48,000 characters | Same |
| Evidence selection | 8 windows × 4,000 characters; 28,000-character cap | Same, with improved source coverage |
| Candidate count | 3 | 3 |
| Thinking / reasoning effort | Enabled / `medium` | Enabled / `medium` |
| HTTP helper temperature | 0.1 | 0.1 |
| Option verification | Earlier candidate/verification flow | Enabled; 4,096-token cap; one parsing retry |
| Parallel jobs | 4 | 8 |

The historical helper permits a short no-thinking output-recovery request for an empty final channel. The full-run scorer chooses `extract_choice(content) or extract_choice(reasoning)`, with a local choice parser. This is a documented local scoring policy; equivalence to the upstream parser on raw generations has not been established. The [official result aggregation code](https://github.com/THUDM/LongBench/blob/main/result.py) reports accuracy from supplied judgments, but that does not make these locally parsed results an official submission.

The Harness uses multiple candidates, evidence windows and verification calls, whereas Direct uses the full prompt and possible output recovery. This experiment did not impose the V3.2 OOLONG shared cumulative-token and physical-call contract on both methods. Retained counters are not sufficient to claim a comparison at equal total compute, a complete monetary-cost audit, or a latency improvement. The Direct cache also precedes the latest 82-question Harness rerun.

## Early 10-question routing check

| Method | Correct / attempted | All-task accuracy |
|---|---:|---:|
| Direct Qwen | 7/10 | 70.00% |
| Old ReaderAgent | 5/10 | 50.00% |
| Early Harness | 7/10 | 70.00% |

All ten Harness cases used `direct_pass` and reused Direct outputs. This pilot established routing behavior for inputs that fit; it did not test expanded-context accuracy. It used strict final-content scoring, unlike the later full runner's content-or-reasoning parser, and used `xhigh` reasoning effort. Its results must not be used as a like-for-like accuracy comparison across optimization stages.

## What the evidence supports

The historical experiments show that retaining complete sources externally and selecting bounded raw evidence can restore useful task execution when direct full-prompt requests fail. The full benchmark produced **59 additional correct answers**, and the targeted rerun solved **49 prompts rejected by Direct**. That is the measured system-level advantage.

A fresh, frozen LongBench evaluation of current V3.2, with exact per-request token counts, strict final-answer parsing, shared cumulative budgets, independent source selection and same-model baselines, remains to be run. These historical scores do not establish current-version leaderboard rank, model-family generality, equal-budget superiority over RLM/MemGPT, or unlimited accurate context.

## Per-question results: dedicated 82-question rerun

Indices are zero-based offsets in the retained dataset order; IDs identify the original benchmark questions. `—` means no parsed choice. All rows remain in the reported denominator. Complete derived records for both runs are in the [results JSON](../benchmarks/longbench_v2_historical.json).

| Index | ID | Gold | Direct | Harness | Harness route |
|---:|---|:---:|:---:|:---:|---|
| 2 | 66f94f9ebb02136c067c4fde | D | — | B | `raw_source` |
| 7 | 66fa208bbb02136c067c5fc1 | B | — | B | `raw_source` |
| 10 | 66ec56dd821e116aacb1cd0e | C | — | D | `raw_source` |
| 12 | 66ebd55f5a08c7b9b35e0698 | A | — | A | `raw_source` |
| 16 | 66f8c9febb02136c067c4511 | C | D | D | `raw_source` |
| 18 | 66f61d7dbb02136c067c1802 | D | — | D | `raw_source` |
| 28 | 66f40b9c821e116aacb30a99 | C | — | D | `raw_source` |
| 31 | 6724631ebb02136c067d7300 | D | — | D | `raw_source` |
| 33 | 66f40e44821e116aacb30b45 | C | — | D | `raw_source` |
| 35 | 66fa3843bb02136c067c655d | D | — | D | `raw_source` |
| 45 | 66ee4c55821e116aacb21596 | D | — | D | `raw_source` |
| 46 | 66faafd9bb02136c067c74a8 | B | — | B | `raw_source` |
| 57 | 67237d8ebb02136c067d6c06 | C | — | C | `raw_source` |
| 58 | 66f1dac1821e116aacb27df1 | A | — | A | `raw_source` |
| 64 | 67246903bb02136c067d74f2 | A | A | A | `direct_pass` |
| 65 | 66f2a4e3821e116aacb2a49b | B | — | B | `raw_source` |
| 74 | 66f2a255821e116aacb2a20b | D | — | D | `raw_source` |
| 96 | 6703f73cbb02136c067cd74a | B | — | D | `raw_source` |
| 115 | 66ebd2475a08c7b9b35e0155 | D | — | A | `raw_source` |
| 116 | 66fa8ccdbb02136c067c6fa7 | B | — | B | `raw_source` |
| 120 | 66f55828821e116aacb3363e | D | — | D | `raw_source` |
| 124 | 66fa2734bb02136c067c627a | A | — | A | `raw_source` |
| 131 | 66f299b3821e116aacb29cfa | B | — | B | `raw_source` |
| 138 | 66f1cd02821e116aacb27d3f | B | — | B | `raw_source` |
| 140 | 66fa542bbb02136c067c686d | A | — | A | `raw_source` |
| 142 | 670765abbb02136c067d06b4 | B | — | B | `raw_source` |
| 143 | 66f3df1e821e116aacb2f7be | A | — | A | `raw_source` |
| 147 | 66ebd34d5a08c7b9b35e035d | C | — | D | `raw_source` |
| 148 | 66ead7bd5a08c7b9b35dca7a | D | — | D | `raw_source` |
| 154 | 6724c5febb02136c067d78e7 | A | — | B | `raw_source` |
| 157 | 66f3fb96821e116aacb3043f | A | — | — | `raw_source` |
| 159 | 66ed5be2821e116aacb1fb57 | D | — | D | `raw_source` |
| 168 | 66f00f1a821e116aacb252cf | B | — | A | `raw_source` |
| 174 | 66f2a46d821e116aacb2a41a | B | — | B | `raw_source` |
| 191 | 66f2cb0f821e116aacb2ba6d | A | — | D | `raw_source` |
| 193 | 66faa0f5bb02136c067c722c | A | — | A | `raw_source` |
| 198 | 66f954a5bb02136c067c511a | B | — | B | `raw_source` |
| 205 | 66f1ae08821e116aacb27c9c | B | — | D | `raw_source` |
| 212 | 66ebcc335a08c7b9b35df8b6 | D | — | D | `raw_source` |
| 217 | 66ebeb5a5a08c7b9b35e1ba5 | A | — | A | `raw_source` |
| 222 | 670cb73fbb02136c067d2502 | C | — | D | `raw_source` |
| 227 | 6704fe26bb02136c067ce670 | C | — | D | `raw_source` |
| 233 | 67040983bb02136c067cd996 | A | — | B | `raw_source` |
| 251 | 66fa1e4cbb02136c067c5edf | A | — | D | `raw_source` |
| 260 | 66fb75adbb02136c067c7d73 | C | — | D | `raw_source` |
| 262 | 6708ae87bb02136c067d1847 | C | — | B | `raw_source` |
| 263 | 66f3fd3a821e116aacb30533 | B | — | B | `raw_source` |
| 264 | 66ebc04a5a08c7b9b35de740 | B | — | D | `raw_source` |
| 283 | 66ed3960821e116aacb1f616 | C | — | C | `raw_source` |
| 297 | 66f52f0f821e116aacb32ddb | A | — | A | `raw_source` |
| 307 | 66f3a311821e116aacb2de19 | B | D | D | `raw_source` |
| 317 | 66f7f382bb02136c067c3a6c | C | — | C | `raw_source` |
| 328 | 66f2a14b821e116aacb2a128 | C | — | C | `raw_source` |
| 331 | 66f43414821e116aacb310a1 | D | — | D | `raw_source` |
| 335 | 6700d19dbb02136c067cb024 | C | — | A | `raw_source` |
| 336 | 66f3ac0b821e116aacb2e203 | D | — | D | `raw_source` |
| 348 | 66fa69a4bb02136c067c6b75 | D | — | D | `raw_source` |
| 350 | 66fa6702bb02136c067c6abb | B | — | B | `raw_source` |
| 357 | 6707f349bb02136c067d13b9 | C | — | C | `raw_source` |
| 368 | 66fb77e7bb02136c067c7db1 | C | — | D | `raw_source` |
| 369 | 66fa5fbcbb02136c067c6959 | B | — | A | `raw_source` |
| 371 | 66f3e318821e116aacb2f9d1 | B | — | D | `raw_source` |
| 372 | 66fa1b54bb02136c067c5df7 | B | — | B | `raw_source` |
| 390 | 66f2982b821e116aacb29c66 | A | — | A | `raw_source` |
| 394 | 66f11349821e116aacb269e0 | A | — | C | `raw_source` |
| 395 | 66f28794821e116aacb298ee | A | — | A | `raw_source` |
| 405 | 66f3c5f2821e116aacb2eccf | A | — | C | `raw_source` |
| 407 | 67066b87bb02136c067cfaa3 | B | — | A | `raw_source` |
| 410 | 66f24538821e116aacb2865e | D | — | C | `raw_source` |
| 415 | 66f57379821e116aacb33b7a | D | — | D | `raw_source` |
| 423 | 66f3e604821e116aacb2fadf | D | — | D | `raw_source` |
| 427 | 6700eb6ebb02136c067cb152 | D | — | D | `raw_source` |
| 438 | 66ec17e4821e116aacb1a6a7 | D | — | D | `raw_source` |
| 447 | 66fcf80dbb02136c067c928e | B | — | B | `raw_source` |
| 449 | 6724cae7bb02136c067d79be | B | — | D | `raw_source` |
| 453 | 66f40c5e821e116aacb30ad2 | A | — | B | `raw_source` |
| 455 | 66f51ab2821e116aacb325fb | C | — | C | `raw_source` |
| 462 | 66f908e3bb02136c067c4992 | A | — | C | `raw_source` |
| 468 | 67246d0ebb02136c067d75f0 | D | — | D | `raw_source` |
| 470 | 66ed24c5821e116aacb1eee6 | B | — | B | `raw_source` |
| 471 | 66ebdd1c5a08c7b9b35e113b | D | — | D | `raw_source` |
| 486 | 66fa36c9bb02136c067c6517 | C | — | C | `raw_source` |
