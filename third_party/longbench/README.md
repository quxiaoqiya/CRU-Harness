# LongBench v2 attribution

Benchmark: [LongBench v2](https://longbench2.github.io/),
[paper](https://arxiv.org/abs/2412.15204),
[official repository](https://github.com/THUDM/LongBench), and
[dataset](https://huggingface.co/datasets/zai-org/LongBench-v2).

The benchmark adapters use the official zero-shot prompt convention and compare
their compatible answer parser with the official `extract_answer` convention.
The reference `pred.py` file's last-change commit is
`c5ea10bcd06285223c58dfed76bbc92d22273709`; its SHA-256 is
`ab63f77866a1c0dc770582bc3fe6b014c3b4be4667399b0ee267075780c6a138`.
This code reference does not establish a dataset revision.

The included [MIT license](LICENSE) applies to the upstream software and prompt
material. Dataset terms are addressed separately in [DATA_LICENSE.md](../../DATA_LICENSE.md).
Raw benchmark contexts and response archives are not redistributed here.
