# CRU-Harness data and third-party materials

CRU-Harness is a standalone software project. Its published benchmark summary
reports experiments on the external OOLONG-synth, LongBench v2 and LongMemEval-V2 benchmarks;
CRU-Harness does not claim authorship of these benchmarks or their source datasets.

## Included materials

- Project-authored source code, documentation, synthetic examples and tests.
- Derived evaluation results, sample identifiers, version references and hashes
  in `benchmarks/oolong_v32_20261101.json`.
- Derived OpenAI transfer scores, usage, source-envelope counts and audit
  metadata in `benchmarks/openai_transfer_20261102.json` and
  `benchmarks/openai_transfer_integrity_20261102.json`, with independent score
  and cost checks in `benchmarks/openai_transfer_score_audit_20261102.json`.
  These reuse the same
  external OOLONG-synth sources; raw questions, answers and fact ledgers are
  not distributed in these artifacts.
- Derived historical LongBench v2 results, sample identifiers, answer letters,
  source-context hashes, and local artifact hashes in
  `benchmarks/longbench_v2_historical.json`. LongBench v2 originates from
  [Bai et al.](https://arxiv.org/abs/2412.15204); obtain its data from the
  [official dataset](https://huggingface.co/datasets/zai-org/LongBench-v2).
- Derived three-model LongBench capacity and accuracy results, answer letters,
  token accounting, source hashes, and head/tail span metadata in
  `benchmarks/longbench_openai_limits_20261102.json` and its associated audits.
  These preserve a previously consumed 12-source regression cohort; they do
  not redistribute the source contexts, option text, or raw model replies.
- Derived LongMemEval-V2 small-tier pilot scores, source hashes, per-call budget
  receipts and image-binding audit metadata in
  `benchmarks/longmemeval_v2_qwen_20261103.json` and its offline audit. Only 20
  questions and two shared histories were evaluated; raw trajectory text,
  questions, gold, screenshots and model replies are not distributed.
- Derived six-arm LongMemEval-V2 question-holdout metrics, budget receipts,
  code/source hashes and offline replay metadata in
  `benchmarks/longmemeval_v2_visual_holdout_20261103.json` and two associated
  audits. Local screenshot feature vectors are not distributed; only their
  integrity hashes, image counts and rederivation receipt are included.
- Derived five-arm LongMemEval-V2 completion/display question-holdout scores,
  per-call reservations, reversible display/source-position receipts and replay
  metadata in `benchmarks/longmemeval_v2_compact_holdout_20261103.json` and its
  offline audit. Only hashes and numeric source intervals are distributed;
  raw text, edit content, screenshot bytes, private replies and labels stay local.
- Derived LongMemEval-V2 multimodal visual-question evaluation metadata in
  `benchmarks/longmemeval_v2_visual_anchor_20261103.json` and three stage audits:
  12 consumed development questions evaluated with two controller versions,
  then eight new question-disjoint visual questions (104 outcomes total).
  Query-image hashes, scores, budgets and source fingerprints are included;
  raw images, captions, UI observations, actions, references and replies are not.
- The upstream LongMemEval-V2 Apache-2.0 software license and attribution in
  [`third_party/longmemeval_v2/`](third_party/longmemeval_v2/README.md).
- The upstream LongBench software license and prompt/parser attribution in
  [`third_party/longbench/`](third_party/longbench/README.md).
- An unmodified official OOLONG scoring helper and its MIT notice in
  [`third_party/oolong/`](third_party/oolong/README.md).

The repository does not distribute the benchmark source contexts, research
articles, PDFs, figures, model request traces, private deployment configuration
or credentials. Obtain external datasets from their official sources under
their respective terms. The scoring helper's MIT license does not automatically
license the underlying benchmark datasets.

## Software and source-data rights

No project-wide open-source license has been selected for the project-authored
software. This data notice does not grant a software or dataset license.
Third-party materials retain their original authorship and applicable license
notices. Public availability or open-access status alone does not establish
permission to redistribute source documents.

Benchmark provenance, fixed revisions and scoring definitions are documented in
[`docs/benchmark_v32.md`](docs/benchmark_v32.md). Related method attributions are
documented in [`docs/related_work.md`](docs/related_work.md).
The historical LongBench evaluation scope and unrecorded upstream revision are
documented in [`docs/benchmark_longbench_v2.md`](docs/benchmark_longbench_v2.md).
The three-model OpenAI transfer protocol and reused-cohort scope are documented
in [`docs/openai_transfer_20261102.md`](docs/openai_transfer_20261102.md).
The LongBench extreme-context comparison, incomplete first trial, zero-generation
recovery and separate head/tail control are documented in
[`docs/longbench_openai_limits_20261102.md`](docs/longbench_openai_limits_20261102.md).

The LongMemEval-V2 pilot, pinned versions and judge substitution are documented
in [`docs/longmemeval_v2_qwen_20261103.md`](docs/longmemeval_v2_qwen_20261103.md).

The visual retrieval and abstention question holdout, including failed candidates,
is documented in [`docs/longmemeval_v2_visual_holdout_20261103.md`](docs/longmemeval_v2_visual_holdout_20261103.md).

The native completion and reversible-display controls, including failed development
trials and the new question holdout, are documented in
[`docs/longmemeval_v2_compact_holdout_20261103.md`](docs/longmemeval_v2_compact_holdout_20261103.md).

The multimodal visual-question evaluation and its unsuccessful effect-constraint
variant are documented in
[`docs/longmemeval_v2_visual_anchor_20261103.md`](docs/longmemeval_v2_visual_anchor_20261103.md).
