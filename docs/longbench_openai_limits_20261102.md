# LongBench v2: native Direct versus bounded Harness at extreme lengths

**English** | [简体中文](longbench_openai_limits_20261102.zh-CN.md)

This experiment pairs full native-context Direct with bounded-evidence Harness on 12 unpadded natural LongBench sources using three exact models: 72 main logical outcomes plus 3 separate capacity probes. The first 75 terminal records included 39 count-network failures with no generation sent. A separately frozen recovery added 39 execution records for those tasks. A distinct head/tail quality control followed the preregistered length-and-money rule. Labels were first opened only after every original, recovery, and conditional-control outcome was terminal or an explicit budget skip was recorded.

This is a length-biased regression evaluation on previously consumed sources, not a new blind holdout, full 503 score, or official leaderboard result. The arms have different compute budgets; the comparison does not establish same-compute accuracy uplift.

## Original incomplete trial and registered recovery

The first-trial delivery results are preserved below before the recovered main table, so infrastructure recovery is not presented as an algorithmic accuracy gain. The 39 cases were selected only by the frozen generic-network error, empty generation ledger, zero physical generations, and zero token charges. Selection did not use outputs or correctness. No originally generated task was rerun.

| Model | Original Direct correct delivery /12 | Original Harness correct delivery /12 | Registered recovery tasks | Original generation calls | New recovery generation calls |
|---|---:|---:|---:|---:|---:|
| gpt-6-sol | 0/12 (0.00%) | 3/12 (25.00%) | 14 | 24 | 51 |
| gpt-6-luna | 0/12 (0.00%) | 4/12 (33.33%) | 13 | 40 | 37 |
| gpt-6.1-terra | 0/12 (0.00%) | 4/12 (33.33%) | 12 | 55 | 22 |

| Model | Recovered Direct indices | Recovered Harness indices | Recovered probe indices |
|---|---|---|---|
| gpt-6-sol | 35, 147, 335, 449, 453 | 33, 35, 143, 147, 257, 335, 395, 449 | 35 |
| gpt-6-luna | 143, 251, 260, 438, 449, 453 | 35, 143, 251, 395, 438, 453 | 35 |
| gpt-6.1-terra | 33, 35, 143, 147, 257, 395, 449, 453 | 35, 251, 257 | 35 |

The original 75 and new 39 records remain immutable. Recovery added count-only network retries up to 3 attempts with 5/15-second delays and safe numeric error observations. It did not retry generations, modify the 56 original algorithm/controller files, or reset consumed generation budgets. Each selected original task had zero consumption, so parent plus recovery remains within the same logical task budget.

## Recovered main outcomes and capacity denials

| Model | Direct task success /all 12 | Direct local capacity denials | Direct correct delivery /published-admissible | Harness correct /all 12 | Harness above-published-native subgroup |
|---|---:|---:|---:|---:|---:|
| gpt-6-sol | 0/12 (0.00%) | 8 | 0/4 (0.00%) | 8/12 (66.67%) | 6/8 (75.00%) |
| gpt-6-luna | 0/12 (0.00%) | 8 | 0/4 (0.00%) | 6/12 (50.00%) | 5/8 (62.50%) |
| gpt-6.1-terra | 1/12 (8.33%) | 8 | 1/4 (25.00%) | 7/12 (58.33%) | 5/8 (62.50%) |

The Direct/all-12 column measures operational task success. Admission-denied cases have no semantic answer and are not labeled wrong model answers. The published-admissible primary denominator retains provider context rejections, other API failures, missing outputs, and wrong answers. It is correct-delivery rate, not pure semantic accuracy. Harness retains all 12.

### Direct rejection and infrastructure breakdown

| Model | Local admission denials | Provider context rejections | Rejected within published admission | Infrastructure terminal failures | Eligibility unknown | Actual returned choices |
|---|---:|---:|---:|---:|---:|---:|
| gpt-6-sol | 8 | 2 | 2 | 1 | 0 | 1 |
| gpt-6-luna | 8 | 2 | 2 | 0 | 0 | 2 |
| gpt-6.1-terra | 8 | 2 | 2 | 0 | 0 | 2 |

Exact admission under the published 1.05M specification does not establish actual provider acceptance. A context_length_exceeded response within that range retains its original error status and is listed separately, not as a wrong semantic answer, a local denial, or proof that the 1.05M service window works. Rate/quota 429 and uncertain network delivery are separate categories.

## Paired comparison on native-admissible sources

| Model | Primary paired n | Both correct | Direct only | Harness only | Both unsuccessful | Both-answers-available n (secondary) |
|---|---:|---:|---:|---:|---:|---:|
| gpt-6-sol | 4 | 0 | 0 | 2 | 2 | 1 |
| gpt-6-luna | 4 | 0 | 0 | 1 | 3 | 2 |
| gpt-6.1-terra | 4 | 1 | 0 | 1 | 2 | 2 |

Conditioning on both answers being available is secondary, to avoid inflating accuracy by dropping failures. Few sources and one run per arm do not establish stable rankings. Wrong answers, capacity limits, and infrastructure failures are different outcomes.

| Model | Sources actually context-rejected by Direct | Harness correct delivery on those sources |
|---|---:|---:|
| gpt-6-sol | 2 | 1/2 (50.00%) |
| gpt-6-luna | 2 | 0/2 (0.00%) |
| gpt-6.1-terra | 2 | 1/2 (50.00%) |

This table includes actual HTTP context_length_exceeded outcomes only, separate from the above-specification group whose Direct generations were not submitted. The separately registered probes are not scored in this table.

## Observed length frontier

| Model | Largest prompt with an actual Direct answer | Smallest actual Direct context rejection | Largest Harness attempt | Largest with a Harness answer | Largest correct Harness prompt | Longest correct /published-native | Longest correct /16k workspace |
|---|---:|---:|---:|---:|---:|---:|---:|
| gpt-6-sol | 845,093 (index 147) | 997,650 (index 335) | 4,264,583 (index 143) | 4,264,583 (index 143) | 4,264,583 (index 143) | 4.06x | 260.29x |
| gpt-6-luna | 845,093 (index 147) | 997,650 (index 335) | 4,264,583 (index 143) | 4,264,583 (index 143) | 4,264,583 (index 143) | 4.06x | 260.29x |
| gpt-6.1-terra | 845,093 (index 147) | 997,650 (index 335) | 4,264,583 (index 143) | 4,264,583 (index 143) | 4,264,583 (index 143) | 4.06x | 260.29x |

Lengths are complete native-Direct Responses admission inputs, including the resource notice and 16 safety tokens, not raw-text tokenization. The 1,050,000 native window and 16,384 experimental workspace are different denominators. Actual Direct acceptance and rejection describe this sampled interval, not a precise hard limit; the errors reported no numeric maximum. Attempted, answered, and correct sizes are separate. A correct multiple-choice answer does not establish reliable understanding of the entire source or an absolute limit.

Official model-window sources: [gpt-6-sol](https://developers.openai.com/api/docs/models/gpt-6-sol) · [gpt-6-luna](https://developers.openai.com/api/docs/models/gpt-6-luna) · [gpt-6.1-terra](https://developers.openai.com/api/docs/models/gpt-6.1-terra)

## Three separate API capacity probes

| Model | Source index /admitted input tokens | Actual status | Safe HTTP/code | Context maximum reported in generation error | Physical calls | Known tokens | Unknown-usage reserved tokens | Conservative upper cost |
|---|---:|---|---|---:|---:|---:|---:|---:|
| gpt-6-sol | 35 / 1,138,985 | provider_context_rejected | 400/context_length_exceeded | not_reported | 1 | 0 | 1,140,009 | USD 5.710285 |
| gpt-6-luna | 35 / 1,138,985 | provider_context_rejected | 400/context_length_exceeded | not_reported | 1 | 0 | 1,140,009 | USD 0.285514 |
| gpt-6.1-terra | 35 / 1,138,985 | provider_context_rejected | 400/context_length_exceeded | not_reported | 1 | 0 | 1,140,009 | USD 2.855142 |

Probes test provider acceptance/rejection and transport only and receive no accuracy score. Only explicit context_length_exceeded establishes a context rejection; 429 is a rate/quota event, and generic 400 or transport failures do not establish a context boundary. The recovery observer retains whitelisted codes, parameters, and numeric candidates without error prose. Reported maxima are numbers in provider errors, not an independently measured absolute limit. Missing usage retains a full reservation, not known actual inference usage.

## Usage and estimated cost

| Model /arm | Generation calls | Known input | Known output | Unknown-usage calls | Unknown reserved tokens | Ledger error calls | Estimated upper cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| gpt-6-sol / direct | 4 | 845,077 | 11 | 3 | 2,166,905 | 3 | USD 14.790385 |
| gpt-6-sol / harness | 70 | 559,847 | 5,468 | 1 | 9,129 | 1 | USD 1.492480 |
| gpt-6-sol / capacity_probes | 1 | 0 | 0 | 1 | 1,140,009 | 1 | USD 5.710285 |
| gpt-6-sol / headtail_control | 2 | 1,532,419 | 22 | 0 | 0 | 0 | USD 7.662425 |
| gpt-6-luna / direct | 4 | 963,177 | 22 | 2 | 2,047,765 | 2 | USD 0.739011 |
| gpt-6-luna / harness | 72 | 563,898 | 6,438 | 2 | 21,207 | 2 | USD 0.077893 |
| gpt-6-luna / capacity_probes | 1 | 0 | 0 | 1 | 1,140,009 | 1 | USD 0.285514 |
| gpt-6-luna / headtail_control | 2 | 1,532,419 | 22 | 0 | 0 | 0 | USD 0.383121 |
| gpt-6.1-terra / direct | 4 | 963,177 | 25 | 2 | 2,047,765 | 2 | USD 7.390122 |
| gpt-6.1-terra / harness | 72 | 580,977 | 6,129 | 0 | 0 | 0 | USD 0.756866 |
| gpt-6.1-terra / capacity_probes | 1 | 0 | 0 | 1 | 1,140,009 | 1 | USD 2.855142 |
| gpt-6.1-terra / headtail_control | 2 | 1,532,419 | 22 | 0 | 0 | 0 | USD 3.831212 |

The original, recovery, and conditional head/tail stages including probes made 235 generation requests, with 9,091,569 known input/output tokens and 9,712,798 separately reserved unknown-usage tokens. Their formal generation-cost upper estimate is USD 45.974458; adding preflight gives USD 45.975822, within the shared USD 50 global cap. These are conservative estimates, not an invoice.

| Runtime stage | Physical generations | Logical count queries including cache | Physical count HTTP attempts | Count retry attempts | Generation-cost upper estimate |
|---|---:|---:|---:|---:|---:|
| original_infrastructure_incomplete | 119 | 523 | 215 | 0 | USD 9.890960 |
| registered_recovery_new_work | 110 | 445 | 170 | 0 | USD 24.206740 |
| headtail_quality_control | 6 | 24 | 6 | 0 | USD 11.876759 |
| combined | 235 | 992 | 391 | 0 | USD 45.974458 |

This runtime table retains original failed count attempts and every recovery count retry. It excludes the separate selection preflight and the head/tail trimming preparation, which are listed below. The aggregate covers recorded requests in this experiment, not the project's entire counting history. Count requests are not generation-token usage.

| Count-request scope | Physical count HTTP attempts |
|---|---:|
| runtime_original_recovery_and_control | 391 |
| headtail_trimming_preparation | 6 |
| separate_candidate_selection_preflight | 34 |
| all_recorded_count_attempts | 431 |

Before generation, monetary admission reserves the maximum ordinary/cached/default 30-minute-write input price and the output price. Above 272k input, whole-request input/cache prices double and output prices multiply by 1.5. Known usage settles at the admitted envelope; unknown usage retains the reservation. Cache and reasoning subsets are not added twice.

Frozen pricing and specification sources: [gpt-6-sol](https://developers.openai.com/api/docs/models/gpt-6-sol) · [gpt-6-luna](https://developers.openai.com/api/docs/models/gpt-6-luna) · [gpt-6.1-terra](https://developers.openai.com/api/docs/models/gpt-6.1-terra)

Separate from the formal run, three availability probes used 249 input + 15 output = 264 tokens. Counting attempts and the initial field-size failure are separate; they are not generation-token usage.

## Controller-caught internal errors

| Model | Harness terminal errors | Ledger error requests | Caught candidate errors | Caught option-verifier errors | Parse errors among them | Explicit verifier retries |
|---|---:|---:|---:|---:|---:|---:|
| gpt-6-sol | 0 | 1 | 0 | 3 | 0 | 0 |
| gpt-6-luna | 0 | 2 | 0 | 2 | 0 | 0 |
| gpt-6.1-terra | 0 | 0 | 0 | 0 | 0 | 0 |

A controller can still deliver an answer after a candidate or verifier failure, so a clean terminal status does not imply every internal call succeeded. The physical ledger is authoritative; explicit recoveries and retries share request, token, and monetary budgets. One failure can appear in both ledger and caught-error counters, so these are not additive counts of independent failures.

## Per-source results

Each cell is Direct/Harness predicted choice, with correctness marks. Capacity denial is labeled denied, not an incorrect answer. A/B/C/D are short derived benchmark choice letters without option text.

| Index / ID | Domain | Reference | gpt-6-sol | gpt-6-luna | gpt-6.1-terra |
|---|---|---|---|---|---|
| 143 / 66f3df1e821e116aacb2f7be | Code Repository Understanding | A | denied / A ✓ | denied / A ✓ | denied / A ✓ |
| 260 / 66fb75adbb02136c067c7d73 | Code Repository Understanding | C | denied / C ✓ | denied / C ✓ | denied / C ✓ |
| 453 / 66f40c5e821e116aacb30ad2 | Code Repository Understanding | A | denied / B ✗ | denied / B ✗ | denied / B ✗ |
| 449 / 6724cae7bb02136c067d79be | Long Structured Data Understanding | B | denied / B ✓ | denied / B ✓ | denied / C ✗ |
| 438 / 66ec17e4821e116aacb1a6a7 | Multi-Document QA | D | denied / D ✓ | denied / C ✗ | denied / D ✓ |
| 33 / 66f40e44821e116aacb30b45 | Long In-context Learning | C | denied / D ✗ | denied / C ✓ | denied / B ✗ |
| 147 / 66ebd34d5a08c7b9b35e035d | Single-Document QA | C | B ✗ / B ✗ | B ✗ / B ✗ | B ✗ / B ✗ |
| 257 / 6719185cbb02136c067d40ab | Long-dialogue History Understanding | B | missing / B ✓ | A ✗ / B ✓ | B ✓ / B ✓ |
| 395 / 66f28794821e116aacb298ee | Long In-context Learning | A | provider_rejected / D ✗ | provider_rejected / B ✗ | provider_rejected / D ✗ |
| 335 / 6700d19dbb02136c067cb024 | Long Structured Data Understanding | C | provider_rejected / C ✓ | provider_rejected / A ✗ | provider_rejected / C ✓ |
| 35 / 66fa3843bb02136c067c655d | Code Repository Understanding | D | denied / D ✓ | denied / A ✗ | denied / D ✓ |
| 251 / 66fa1e4cbb02136c067c5edf | Code Repository Understanding | A | denied / A ✓ | denied / A ✓ | denied / A ✓ |

## Provenance, protocol, and interpretation

Sources: the [official LongBench v2 project](https://longbench2.github.io/), [official code](https://github.com/THUDM/LongBench), and [official dataset](https://huggingface.co/datasets/zai-org/LongBench-v2). The HF split is named train. The original cache did not record an upstream revision; the report retains null and publishes cache/input/per-source hashes rather than inventing a revision.

Native Direct has a 1,050,000 per-request/cumulative ceiling, one generation, and 1,024 output ceiling. Harness has 16,384 per request, 65,536 cumulative, 32 generations, and 2,048 output per call. Three evidence views and support/against/unknown option verification share the ledger. All models use reasoning=none, store=false, truncation=disabled, default tier, with no temperature or seed sent. There are no implicit retries or reused Direct answers.

Preflight found a 10,485,760-character single-string field limit. Longer text becomes ordered input_text parts within the same message, each at most 10,000,000 characters; concatenation exactly preserves the original prompt. Counting and generation use the same lossless encoding. This repairs a transport field limit, not native context capacity. No source was artificially repeated or padded.

This uses the LongBench multiview-evidence/option-verification path, distinct from OOLONG V3.2 structured-fact reduction. Full sources remain external and the model receives bounded evidence; a correct answer does not mean the model read every source token. Twelve length-biased sources, a single generated outcome per logical arm, and multiple-choice chance limit quality-frontier conclusions.

Freeze SHA-256: 941c86120e66020439571fbe8888999c643ebcf83fc315d7ec7e1e0e1045208d
Recovery freeze SHA-256: febefea46171ba7d843104aa8857bff82670b597e6b0bb0de6b640e7c7f312c1

The [public JSON](../benchmarks/longbench_openai_limits_20261102.json) contains 56 original algorithm source fingerprints, recovery and head/tail wrapper fingerprints, all 75 original outcomes, the 39-case recovery lineage, 72 recovered main logical outcomes, 3 probes, 6 auxiliary outcomes, exact counts, short choice letters, known/unknown usage, and cost definitions. Raw questions, contexts, option text, evidence windows, model replies, credentials, and private paths are not distributed.

A standard-library public verifier checks short-choice scoring, cohort matrices, phase usage and cost conservation, source fingerprints, and private-field exclusions. It makes no API calls and reads no private labels. This is not a replay of unpublished API responses or access logs.

```bash
python3 -S scripts/verify_longbench_openai_limits_results.py
```

## Supplemental quality control: lossy head/tail Direct

Before labels opened, the remaining-money rule selected the fixed longest sources 143 and 260 when affordable, only 143 otherwise, or an explicit budget skip. Each source permits one generation, an 800,000 input-plus-output ceiling, and 1,024 output. The same character-balanced prefix/suffix and source spans are shared across models. This is a distinct quality control with a larger input budget than Harness, not full-source reading or a same-compute comparison.

| Model | Planned sources | Head/tail compatible correct | Head/tail strict actual-text correct | Harness correct on the same full sources | Parseable head/tail outputs |
|---|---:|---:|---:|---:|---:|
| gpt-6-sol | 2 | 2/2 (100.00%) | 2/2 (100.00%) | 2/2 (100.00%) | 2 |
| gpt-6-luna | 2 | 1/2 (50.00%) | 1/2 (50.00%) | 2/2 (100.00%) | 2 |
| gpt-6.1-terra | 2 | 2/2 (100.00%) | 2/2 (100.00%) | 2/2 (100.00%) | 2 |

The primary quality comparison uses the same local compatible choice extraction as the main experiment. The strict column strips stars from actual head/tail model text and applies the two capitalized The correct answer is patterns; bare A is not automatically accepted. The main Harness result is its controller choice, which is not reinterpreted as a strict score of raw model text. The parser scopes are separate.

| Source index | Original characters | Retained characters /fraction | Retained source spans | Dropped middle span | Prepared admission-input tokens by model |
|---|---:|---:|---|---|---|
| 143 | 16,182,936 | 2,997,835 / 18.52% | [[0, 1498918], [14684019, 16182936]] | [1498918, 14684019] | gpt-6-sol: 786,193, gpt-6-luna: 786,193, gpt-6.1-terra: 786,193 |
| 260 | 11,918,963 | 2,639,965 / 22.15% | [[0, 1319983], [10598981, 11918963]] | [1319983, 10598981] | gpt-6-sol: 746,258, gpt-6-luna: 746,258, gpt-6.1-terra: 746,258 |

No characters are inserted: retained prefix and suffix are concatenated and the middle is explicitly discarded. Original, trimmed-text, and complete trimmed-prompt hashes are in the JSON. This follows the upstream head/tail idea with character spans and exact whole-request counting; it is not a bit-identical reproduction of upstream token-half selection.

| Source /model | Reference | Compatible prediction | Strict prediction | Actual status | Known tokens | Unknown reserved tokens | Upper estimated cost |
|---|---|---|---|---|---:|---:|---:|
| 143 / gpt-6-sol | A | A | A | returned | 786,188 | 0 | USD 3.931050 |
| 143 / gpt-6-luna | A | A | A | returned | 786,188 | 0 | USD 0.196552 |
| 143 / gpt-6.1-terra | A | A | A | returned | 786,188 | 0 | USD 1.965525 |
| 260 / gpt-6-sol | C | C | C | returned | 746,253 | 0 | USD 3.731375 |
| 260 / gpt-6-luna | C | B | B | returned | 746,253 | 0 | USD 0.186569 |
| 260 / gpt-6.1-terra | C | C | C | returned | 746,253 | 0 | USD 1.865688 |

The upstream code is pinned at [c5ea10bcd06285223c58dfed76bbc92d22273709](https://raw.githubusercontent.com/THUDM/LongBench/c5ea10bcd06285223c58dfed76bbc92d22273709/pred.py). This is a code pin, not a dataset revision. Only one or two sources, one generation, and the discarded middle limit the inference; these results are not pooled into the main 12-source score.

Head/tail freeze SHA-256: dfb25fdb5cd26067a881dca5747b78ded10c7a8e5389091eacb2529a26890c3f

## Post-run trajectory and selected-raw-reply diagnostics

The independent [cost and score audit](../benchmarks/longbench_openai_score_audit_20261102.json) checks phase budgets, model identity, known/unknown usage, costs, and primary scoring. The separate [trajectory audit](../benchmarks/longbench_openai_trajectory_audit_20261102.json) checks source bindings, request hashes, and replayable controller paths. Both ran offline after final scoring, with no additional model or count API calls. Trajectory replay reuses frozen production selection, parsers, and controller; it does not independently verify semantic entailment or reference-answer truth.

| Scope | Records | Complete trajectory replay passed | Trajectory unavailable |
|---|---:|---:|---:|
| original + recovery + auxiliary history | 120 | 80 | 40 |
| main logical outcomes including capacity probes | 75 | 74 | 1 |
| auxiliary head/tail | 6 | 6 | 0 |

All 120 token/money ledgers pass, but not all trajectories replay completely. The 40 unavailable trajectories comprise 39 preserved original zero-generation count failures and gpt-6-sol Harness index 251. That original task made four generation attempts and lacks an original count-exception payload, so its complete trajectory and selected raw reply cannot be confirmed. Its primary recorded answer remains correct and unchanged; a passing ledger does not replace missing trajectory evidence.

Strict-format diagnostics apply only to the complete raw reply originally selected, never to a candidate reselected by strict extraction. Of 81 logical receipts including three unscored probes, 46 raw replies are traceable: 35 Harness, five native Direct, and six head/tail. All 46 match the upstream strict patterns with zero disagreement from the recorded controller choices. Missing raw replies are unavailable, not format failures. The table is a retrospective sensitivity analysis on traceable subsets; it does not replace primary accuracy or constitute an official leaderboard score.

| Model /arm | Planned | Traceable raw replies | Strict format accepted | Strict correct /traceable subset | Unavailable raw replies | Primary correct /all planned |
|---|---:|---:|---:|---:|---:|---:|
| gpt-6-sol / direct | 12 | 1 | 1 | 0/1 (0.00%) | 11 | 0/12 (0.00%) |
| gpt-6-sol / harness | 12 | 11 | 11 | 7/11 (63.64%) | 1 | 8/12 (66.67%) |
| gpt-6-sol / headtail_control | 2 | 2 | 2 | 2/2 (100.00%) | 0 | 2/2 (100.00%) |
| gpt-6-luna / direct | 12 | 2 | 2 | 0/2 (0.00%) | 10 | 0/12 (0.00%) |
| gpt-6-luna / harness | 12 | 12 | 12 | 6/12 (50.00%) | 0 | 6/12 (50.00%) |
| gpt-6-luna / headtail_control | 2 | 2 | 2 | 1/2 (50.00%) | 0 | 1/2 (50.00%) |
| gpt-6.1-terra / direct | 12 | 2 | 2 | 1/2 (50.00%) | 10 | 1/12 (8.33%) |
| gpt-6.1-terra / harness | 12 | 12 | 12 | 7/12 (58.33%) | 0 | 7/12 (58.33%) |
| gpt-6.1-terra / headtail_control | 2 | 2 | 2 | 2/2 (100.00%) | 0 | 2/2 (100.00%) |

The denominators represent different subsets. Sol Harness has 7/11 correct among strict-traceable replies and 8/12 primary correct because index 251 lacks a traceable raw reply; this is not evidence of a formatting failure. Receipts publish only short choices, raw-reply hashes, source hashes, and selected positions, without model text.
