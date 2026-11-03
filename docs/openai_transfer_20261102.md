# CRU-Harness V3.2 OpenAI transfer evaluation

**English** | [简体中文](openai_transfer_20261102.zh-CN.md)

The same frozen V3.2 algorithm was replayed on the 12 previously used OOLONG-synth cohort-5 sources with three specified OpenAI models. All 36 model/sample pairs became terminal before unified scoring. This is a transfer replay, not a new blind holdout; there is no direct-generation comparison arm.

## Results by model

| Model | Full /12 | Full-credit rate | Official mean /100 | Strict EM /12 | Nonempty | Errors | Descriptive Wilson 95% context-unit interval |
|---|---:|---:|---:|---:|---:|---:|---|
| gpt-6-sol | 9 | 75.00% | 75.00 | 5 | 12/12 | 0 | 46.8%–91.1% |
| gpt-6-luna | 6 | 50.00% | 50.00 | 4 | 9/12 | 3 | 25.4%–74.6% |
| gpt-6.1-terra | 10 | 83.33% | 83.33 | 5 | 12/12 | 0 | 55.2%–95.3% |

The official mean includes numeric partial credit `0.75 ** abs(error)` and is not an accuracy percentage. Failed, empty and unknown outcomes remain in each model’s denominator of 12. The 12 distinct source contexts are reused across models to form 36 pairs. Distinct IDs and hashes do not establish statistical independence; underlying record overlap is not excluded. Wilson intervals are descriptive context-unit summaries and do not confirm stable superiority.

## Usage and estimated cost

| Model | Generation calls | Known input tokens | Known output tokens | Charged/reserved tokens | Unknown-usage requests | Estimated upper USD | Known-usage base I/O estimate USD |
|---|---:|---:|---:|---:|---:|---:|---:|
| gpt-6-sol | 119 | 265,864 | 39,791 | 305,655 | 0 | 3.056550 | 0.929638 |
| gpt-6-luna | 82 | 194,168 | 38,636 | 232,804 | 0 | 0.116402 | 0.038735 |
| gpt-6.1-terra | 118 | 265,532 | 39,757 | 305,289 | 0 | 1.526445 | 0.464317 |

The formal generation-cost conservative estimate totals **$4.699397**, under the frozen $15 cap. It is not an invoice. Each call reserves `(admitted input + reserved output) × r_max` before generation; known usage settles actual input/output in the same envelope, while uncertain delivery retains the full reservation. Base I/O estimates cover known usage only. Cache, cache-write and reasoning details are not added twice. The 16k request limit is below the 272k long-input pricing threshold; nondefault tiers or other pricing deviations must be reported separately.

Official sources for Standard prices, the long-input threshold and the frozen per-token price envelope: [gpt-6-sol](https://developers.openai.com/api/docs/models/gpt-6-sol) · [gpt-6-luna](https://developers.openai.com/api/docs/models/gpt-6-luna) · [gpt-6.1-terra](https://developers.openai.com/api/docs/models/gpt-6.1-terra)

The separate minimal API-availability preflight used 33 input + 15 output = 48 tokens and is excluded from the formal scores and costs above. Token-count requests are audited separately and are not generation calls.

## Model identity and counting

| Requested model | Actual model set | Returned tier set | Counting API requests | Counting cache hits |
|---|---|---|---:|---:|
| gpt-6-sol | gpt-6-sol | default | 188 | 480 |
| gpt-6-luna | gpt-6-luna | default | 117 | 328 |
| gpt-6.1-terra | gpt-6.1-terra | default | 181 | 476 |

| Model | Provider response_status counts | Provider incomplete_reason counts |
|---|---|---|
| gpt-6-sol | completed: 119 | none |
| gpt-6-luna | completed: 79, incomplete: 3 | max_output_tokens: 3 |
| gpt-6.1-terra | completed: 118 | none |

`incomplete/max_output_tokens` is a received but unfinished response and can occur when the last output reservation is constrained by the remaining budget. Known actual input/output still enters the ledger. It is distinguished from HTTP failures, unknown usage and an incorrect semantic answer; it does not by itself diagnose model understanding.

## Full-source and direct-capacity controls

| Model | Source-envelope token range | Direct admission-input range | Native-window fits | 16k fits | 65k cumulative fits | Complete measurements |
|---|---:|---:|---:|---:|---:|---:|
| gpt-6-sol | 651,081–1,582,486 | 651,263–1,582,625 | 6/12 | 0/12 | 0/12 | 12/12 |
| gpt-6-luna | 651,081–1,582,486 | 651,263–1,582,625 | 6/12 | 0/12 | 0/12 | 12/12 |
| gpt-6.1-terra | 651,081–1,582,486 | 651,263–1,582,625 | 6/12 | 0/12 | 0/12 | 12/12 |

The published native window of **1,050,000 tokens** is separate from the actively imposed 16,384 per-call and 65,536 cumulative limits. Direct admission uses the full prompt (including the 16-token safety margin) plus its actual reserved output **1,024**, not an assumed 2,048-token output cap. Counts describe complete Responses input envelopes, unlike historical Qwen raw-text counts. No direct generation was run: admission rejection is not a wrong answer and does not imply 0% direct accuracy. Model weights and attention windows are unchanged.

Native-window specifications for the three exact model IDs: [gpt-6-sol](https://developers.openai.com/api/docs/models/gpt-6-sol) · [gpt-6-luna](https://developers.openai.com/api/docs/models/gpt-6-luna) · [gpt-6.1-terra](https://developers.openai.com/api/docs/models/gpt-6.1-terra)

### Descriptive above-native subgroup

Defined per model by `direct admission input + recorded 1,024 output reserve > 1,050,000`, this is a capacity-defined subgroup, separate from all 12 tasks. Qwen expansion ratios are not transferred, and no direct wrong-answer claim is made.

| Model | Subgroup n | Full credit | Official mean /100 | Longest full-credit source envelope | Maximum subgroup source envelope |
|---|---:|---:|---:|---:|---:|
| gpt-6-sol | 6 | 4 | 66.67 | 1577931 | 1582486 |
| gpt-6-luna | 6 | 2 | 33.33 | 1569097 | 1582486 |
| gpt-6.1-terra | 6 | 5 | 83.33 | 1577931 | 1582486 |

## Per-source score calls and tokens

Each cell is official score/100 · generation calls · charged/reserved tokens. Very small positive scores may display as 0.00; the JSON retains their exact values.

| Sample ID | gpt-6-sol | gpt-6-luna | gpt-6.1-terra |
|---|---:|---:|---:|
| oolong-synth-test-920090057 | 100.00 · 24 · 65,046 | 100.00 · 22 · 65,148 | 100.00 · 24 · 65,239 |
| oolong-synth-test-520050245 | 100.00 · 20 · 65,218 | 0.00 · 2 · 6,576 | 100.00 · 20 · 65,294 |
| oolong-synth-test-420040065 | 0.00 · 27 · 65,089 | 0.00 · 9 · 32,154 | 0.00 · 27 · 65,084 |
| oolong-synth-test-220020070 | 100.00 · 1 · 1,478 | 100.00 · 1 · 1,478 | 100.00 · 1 · 1,473 |
| oolong-synth-test-220020041 | 100.00 · 2 · 5,915 | 100.00 · 3 · 6,579 | 100.00 · 2 · 5,915 |
| oolong-synth-test-820080226 | 100.00 · 2 · 2,541 | 100.00 · 3 · 4,176 | 100.00 · 2 · 2,541 |
| oolong-synth-test-621060211 | 0.00 · 26 · 65,327 | 0.00 · 23 · 65,348 | 100.00 · 25 · 64,778 |
| oolong-synth-test-821080221 | 0.00 · 10 · 25,489 | 0.00 · 11 · 40,255 | 0.00 · 10 · 25,423 |
| oolong-synth-test-721070068 | 100.00 · 2 · 3,102 | 100.00 · 2 · 3,102 | 100.00 · 2 · 3,102 |
| oolong-synth-test-721070053 | 100.00 · 1 · 1,466 | 100.00 · 1 · 1,466 | 100.00 · 1 · 1,461 |
| oolong-synth-test-921090209 | 100.00 · 2 · 2,438 | 0.00 · 3 · 3,976 | 100.00 · 2 · 2,433 |
| oolong-synth-test-621060242 | 100.00 · 2 · 2,546 | 0.00 · 2 · 2,546 | 100.00 · 2 · 2,546 |

## Plan and stopping diagnostics

| Model | Plan-validation errors | Plan repairs | No-fact-gain stops | One unresolved record at stop | Recorded final fact losses |
|---|---:|---:|---:|---:|---:|
| gpt-6-sol | 0 | 0 | 0 | 0 | 0 (12/12 measured) |
| gpt-6-luna | 0 | 0 | 2 | 0 | 0 (9/12 measured) |
| gpt-6.1-terra | 0 | 0 | 0 | 0 | 0 (12/12 measured) |

These diagnostics use plan types, aggregate unknown counts and stop counters only, without labels, candidate values or record text. One unresolved record is not proof of a localized semantic error; calculation closure does not independently verify semantic truth. Per-task error types, counting failures and safe HTTP codes are in the JSON.

## Frozen protocol and interpretation

Responses uses reasoning=none, store=false, truncation=disabled and requested default service tier. Temperature and seed are not sent; determinism is not claimed. Each model/sample has its own shared 16k/65k/32-call/2k-output ledger. These sources were already used in the V3.2 experiment; the transfer does not tune against gold. Only the transport and monetary bridge were added.

No new direct, RLM, MemGPT or RAG arm was run. This design cannot estimate causal Harness uplift, establish general cross-model superiority or unlimited accurate context, and is not an official leaderboard submission. Historical Qwen scores are not pooled into this denominator. Pretraining exposure and overlap among underlying records remain unexcluded.

Freeze SHA-256: `7be98f287df0e854f6753a5811071009518e1c250b8d56cf43c33464f2e112c9`

Source IDs/hashes, dataset revisions, official scorer commit, original algorithm and bridge source hashes, all 36 whitelisted results, measurements and cost definitions are in the [public JSON](../benchmarks/openai_transfer_20261102.json). Questions, answers, fact ledgers, raw responses and private paths are not distributed.

[OOLONG](https://github.com/abertsch72/oolong) · [Dataset](https://huggingface.co/datasets/oolongbench/oolong-synth) · [Token counting](https://developers.openai.com/api/docs/guides/token-counting)

## Independent mechanism replay

The [public integrity audit](../benchmarks/openai_transfer_integrity_20261102.json) checks all **36/36 token ledgers**. It replays **33/33 available mechanism artifacts**, with no violations or final accepted-fact losses. The three Luna terminal-error artifacts have no returned mechanism result and are **unavailable**, not passes.

| Model | Available mechanism passes | Unavailable terminal errors | Evidence complete / incomplete |
|---|---:|---:|---:|
| gpt-6-sol | 12/12 | 0 | 7 / 5 |
| gpt-6-luna | 9/9 | 3 | 6 / 3 |
| gpt-6.1-terra | 12/12 | 0 | 8 / 4 |

The replay checked 250 raw classification replies, 14,419 body bindings, 108,732 source references / retained physical records, and 95,530 exact-body transfers. These counts sum model-specific artifacts; they are not cross-model deduplicated records. All three conditional-group artifacts passed independent reduction checks. Production parsers are partly shared with the auditor; this is not an entirely separate parser implementation. Provenance, conservation, coverage, and semantic correctness remain different properties.

## Failure patterns and the transfer result

- `420040065` failed on all three models. The conditional-group task remained largely unclassified: Sol and Terra each retained 1,576 known and 20,551 unknown scope records at budget exhaustion. Luna retained 192 known records before a no-fact-gain stop. Fixed-budget evidence acquisition remains a bottleneck.
- `821080221` also received no full credit on any model. Terra had 39,056 known and zero unknown records but still a wrong numerical answer; Sol had 38,977 known and 79 unknown records. Complete coverage and correct ledger settlement do not establish correct semantic classification or query interpretation.
- Luna's three incomplete responses comprise two calls at the full 2,048-output limit and one with only 818 output tokens reserved near the task ceiling. The remaining Luna errors include a no-fact-gain stop and a wrong answer despite complete reported coverage. Increasing output capacity alone would not establish semantic accuracy.

The experiment provides concrete evidence that the unchanged Harness can execute through another provider and deliver the reported task results. Terra has the highest point score here. The set is too small and reused to establish a stable ranking or general superiority. The next unvalidated engineering targets are output-size-aware batches and independently checked semantic judgments; neither was applied during this run.

## Code and reproduction prerequisites

The new [Responses adapter](../long_context_agent/openai_responses_transport.py) and [frozen transfer runner](../scripts/benchmark_openai_transfer.py) were covered by **54 targeted offline checks before inference**. The [archival exporter](../scripts/export_openai_transfer_report.py) produced the public metrics after unified scoring. A [credential-free config template](../benchmarks/openai_transfer_config.example.json) and [sanitized V3.2 algorithm reference](../benchmarks/openai_transfer_v32_reference.json) are provided.

Inference requires the original local `inputs.jsonl`, `manifest.json`, and `labels.json` artifacts named in the template. These source-bearing artifacts are not bundled. Obtain the pinned upstream dataset and reconstruct or supply artifacts matching the published hashes; this checkout is not a one-command dataset reconstruction. The bundled scorer is at `third_party/oolong/eval_helpers_original.py`. The strict runner uses macOS `sandbox-exec` to deny label reads during inference.

The sanitized reference contains the original algorithm hashes, rather than the full private-path reference freeze. A reproduction using it records a new reference hash and a new run freeze; it does not impersonate the recorded formal freeze. With the required inputs available locally:

```bash
python -m scripts.benchmark_openai_transfer freeze --config benchmarks/openai_transfer_config.example.json
# Set OPENAI_API_KEY locally; never add it to the config or repository.
python -m scripts.benchmark_openai_transfer run --config benchmarks/openai_transfer_config.example.json --execute-frozen-run
python -m scripts.benchmark_openai_transfer score --config benchmarks/openai_transfer_config.example.json
```

These commands perform a new paid run only when explicitly invoked. The archival exporter checks the original run identity and is not a generic exporter for a new freeze.

Verify public result arithmetic and fingerprints without raw data, credentials, or network:

```bash
python3 -S scripts/verify_openai_transfer_results.py
```

This checks public score aggregates, usage, capacity subgroups, integrity summaries, and source fingerprints. It does not replace scoring replay against raw outputs and labels.

The [independent scoring and cost audit](../benchmarks/openai_transfer_score_audit_20261102.json) re-executed the pinned official scorer and local parser for all 36 outcomes. Every result matched; 319 monetary reservations, actual model IDs, and service tiers passed inspection. This audit is separate from the mechanism replay above.
