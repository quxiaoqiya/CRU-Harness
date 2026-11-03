# CRU-Harness data and third-party materials

CRU-Harness is a standalone software project. Its published benchmark summary
reports experiments on the external OOLONG-synth and LongBench v2 benchmarks;
CRU-Harness does not claim authorship of these benchmarks or their source datasets.

## Included materials

- Project-authored source code, documentation, synthetic examples and tests.
- Derived evaluation results, sample identifiers, version references and hashes
  in `benchmarks/oolong_v32_20261101.json`.
- Derived historical LongBench v2 results, sample identifiers, answer letters,
  source-context hashes, and local artifact hashes in
  `benchmarks/longbench_v2_historical.json`. LongBench v2 originates from
  [Bai et al.](https://arxiv.org/abs/2412.15204); obtain its data from the
  [official dataset](https://huggingface.co/datasets/zai-org/LongBench-v2).
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
