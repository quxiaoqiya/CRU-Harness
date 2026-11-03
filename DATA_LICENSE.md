# CRU-Harness data and third-party materials

CRU-Harness is a standalone software project. Its published benchmark summary
reports experiments on the external OOLONG-synth and LongBench v2 benchmarks;
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
