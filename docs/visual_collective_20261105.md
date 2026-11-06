# Visual evidence beyond native context: frozen results and pixel audit

2026-11-05 · `Qwen3.8-Flash-Next-FP8` · local Mac orchestration.

[Results JSON](../benchmarks/visual_collective_20261105.json) · [51 outcomes](../benchmarks/visual_collective_20261105.csv) · [2,510 page diagnostics](../benchmarks/visual_collective_20261105_pages.csv) · [726 physical receipts](../benchmarks/visual_collective_20261105_requests.csv) · [Source fingerprints](../benchmarks/visual_collective_20261105_sources.json) · [API](visual_page_memory.md)

## Result and scope

Coverage retrieval and external source-bound integer facts let the model process original images whose combined relevant tokens exceed one native request. On seven new visual QA questions, new page v2 scores 4/7 versus Direct 1/7 and old page memory 3/7. On the separately frozen 502-page census, new count memory v2.1 emits facts for every page and scores 72.91% per-page exact, but its total 904 differs from 952. **No method answers the census total exactly.**

A later pixel audit found **one cross-filename duplicate** in the 502-page cohort. The original run stays intact. A **post-hoc, zero-generation slice of 501 pixel-disjoint pages** gives 72.85% per-page exact and 5.05% total error for v2.1, versus 60.28% and 31.34% for 512px Direct. It is not a second independent pre-registered generation experiment.

## Sources, selection and budget

- Visual QA: [MMLongBench-Doc](https://huggingface.co/datasets/yubo2333/MMLongBench-Doc), revision `2ff6aa9237fc777b6627dc57a486e9225ac5fb86`. Seven target PDFs outside the previously used 47-document corpus, four color questions and three counts. Source Git/LFS bytes are pinned. A mismatching PDF was excluded before inference. No answer column was read for selection.
- Global census: [DocLayNet 1.0](https://github.com/DS4SD/DocLayNet), [official ZIP](https://codait-cos-dax.s3.us.cloud-object-storage.appdomain.cloud/dax-doclaynet/1.0.0/DocLayNet_core.zip), ETag `52242220284e12538949c5b06eb2777c-448`. Human Picture/Table annotations define a derived additive task. This is not the original layout-detection benchmark. ZIP entries are range-downloaded and CRC32/SHA256 verified.
- Initial test selection: 2,100 positive pages selected by fixed hash order, plus six separate validation pages. Every global test page has a positive human Picture or Table count; no ranking by count magnitude. Boxes/counts do not enter inference. The additional 502 pages use another fixed hash order from 821 eligible remaining source filenames.
- C = 262,144, using the existing physical boundary calibration and current metadata/tokenizer checks. Original 1025×1025 PNG costs 1,026 image/marker tokens per page. Message/text overhead is separate. Real 1/2/6-image counts match the component accounting.
- Shared ceilings: C/request, 12C = 3,145,728/task, 160 generations, 16,384 output maximum, temperature 0, seed 20261105. Actual compute differs. Old page memory keeps its original 24-tool-execution strategy limit; v2.1 batches 16 pages with 2,048 output tokens. These controller limits confound algorithm-only attribution.
- All original 46 outcomes, five additional outcomes and zero-call structure-recovery outputs were fixed and hashed before reference access. v2.1 was developed from older **response structures**, not the new-page reference counts. A trusted Python label-open guard is not an OS-level security sandbox.

All global pages are required after filename filtering. The 502 original images alone use 515,052 tokens > C, with only 502 image parts, below the service's 999-image cap. The 501-page slice has 514,026 original-image tokens = 1.961C. These are full-page evidence measurements; **minimal sufficient oracle crops have not been proved to exceed C**.

## New visual QA at about 8C source size

| Method | Correct | Page exposure | Calls | Tokens |
|---|---:|---:|---:|---:|
| Original-resolution Direct | 1/7 (14.29%) | 28.57% | 7 | 1,794,153 |
| Text memory | 0/7 (0.00%) | 0.00% | 125 | 1,670,815 |
| Old page Harness | 3/7 (42.86%) | 42.86% | 79 | 1,268,166 |
| Page Harness v2 | 4/7 (57.14%) | 100.00% | 55 | 744,530 |

| QA index | Direct | Text | Old | New v2 |
|---|---|---|---|---|
| 15 | Wrong | Wrong | Correct | Correct |
| 164 | Wrong | Wrong | Wrong | Correct |
| 407 | Wrong | Wrong | Wrong | Wrong |
| 549 | Correct | Wrong | Correct | Correct |
| 625 | Wrong | Wrong | Wrong | Wrong |
| 782 | Wrong | Wrong | Correct | Correct |
| 844 | Wrong | Wrong | Wrong | Wrong |

Page exposure means reference source pages were actually included in completed generations, not that every object was perceived. QA uses a method-blind Qwen semantic-equivalence judge as a local substitute, not an official score. New v2 has zero empty answers and two UNKNOWNs; old visual memory also has zero empty answers, while text memory has two. This cohort alone does not establish an empty-answer-rate improvement over old visual memory.

Questions 407/625 expose the reference pages but fail complete count facts and abstain. Question 844 requires cross-page distinct people: v2 wrongly sums page counts (13 versus reference 7). v2.1 rejects `distinct`/`unique` requests; it has not solved identity deduplication. QA782's printed page 8 is physical PDF page 10; the evidence mapping was corrected before inference, with the answer reference untouched. QA15 says Europe although the map is US; the upstream wording was retained. Seven questions do not establish general or statistically significant superiority.

## Related evidence at 1C, 2C, 4C and 8C

| Nominal scale | Pages | Related image tokens | Direct page exposure | v2 page exposure |
|---|---:|---:|---:|---:|
| 1C | 249 | 255,474 | 100.00% | 100.00% |
| 2C | 502 | 515,052 | 49.60% | 100.00% |
| 4C | 1,009 | 1,035,234 | 24.68% | 100.00% |
| 8C | 2,023 | 2,075,598 | 12.31% | 100.00% |

The four scales are nested, correlated sources. Frozen v2 sees every original page but fails complete fact extraction at all scales and returns UNKNOWN. Additional lossless normalization of already paid responses still leaves the 8C task incomplete (58.87% fact coverage); it does not turn it into a correct answer. **v2.1 accuracy at 4C/8C is not validated.**

A 2,023-image 256px native control was rejected by the tokenizer's 999-image limit before generation: no call and semantic score N/A. A stronger 2×2 montage packs all 2,023 pages into 506 images using 320px tiles, within C including reserved output; that generation also fails a complete answer. It is retained rather than omitted.

## Additional 502-page confirmation, before pixel exclusion

| Method | Predicted total / reference | Page exposure | Fact coverage | Per-page exact | Calls | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| Original-resolution Direct | Incomplete / 952 | 49.60% | 0.00% | 0.00% | 1 | 261,918 |
| All-page 512px Direct | 654 / 952 | 100.00% | 100.00% | 60.36% | 1 | 142,406 |
| Original-resolution count memory v2.1 | 904 / 952 | 100.00% | 100.00% | 72.91% | 32 | 545,324 |
| Old page Harness | Incomplete / 952 | 4.78% | 0.00% | N/A | 18 | 324,849 |
| Text memory | Incomplete / 952 | 0.00% | 0.00% | N/A | 10 | 31,795 |

Per-page exact requires both Picture and Table counts to match; missing facts count wrong when the method has an explicit extraction contract. Scalar-only methods have N/A. All five strict global exact scores are **0/1**. The new protocol uses explicit IDs, exactly-once facts, persistent checkpoints, bounded rereads of only missing/conflicting sources, and deterministic local addition. It completes in 32 calls with no format-recovery call needed on this cohort.

## Pixel duplicate erratum and 501-page slice

Confirmation `c0024` and initial test `h1967` have different source filenames but identical PNG SHA256:

`5405019d57fd1e2ee05d12cd026cde62fccf71b55e4ca8c710f5474a3d62ba90`

The original local generated report's zero-overlap statement refers only to filenames and is incorrect if read as pixel isolation. The public release corrects it explicitly. Removing that page by **pixel identity, not method correctness**, leaves 501 sources with no pixel overlap against the initial 2,106 test/development page records, and no duplicates within the new cohort. The slice is computed from already fixed per-page facts. Missing facts are not filled with zero; no 501-page generation is claimed.

| Method | Page exposure | Per-page exact | Derived subtotal / reference | Relative total error |
|---|---:|---:|---:|---:|
| Original-resolution Direct | 49.50% | 0.00% | Incomplete / 951 | N/A |
| All-page 512px Direct | 100.00% | 60.28% | 653 / 951 | 31.34% |
| Original-resolution count memory v2.1 | 100.00% | 72.85% | 903 / 951 | 5.05% |
| Old page Harness | 4.59% | N/A | Incomplete / 951 | N/A |
| Text memory | 0.00% | N/A | Incomplete / 951 | N/A |

Costs remain those of the full 502-page generation, without an invented pro-rata discount. Original-resolution v2.1 uses about 3.83× the tokens and 32 calls versus one call for compressed Direct. Thus the gain is a resolution/compute/accuracy tradeoff under equal ceilings.

Of 501 pages, v2.1 gets both fields right on 365, Picture right on 401 and Table right on 444. Page totals undercount on 88, overcount on 36 and match on 377; some matching totals hide classification offsets. The final net deficit is 48. Deterministic addition is correct, but the input observations still contain visual errors and human/model block-grouping differences, including small logos. Full source coverage is not object recall.

## Physical cost and verification

| Purpose | Generations | Input + output tokens |
|---|---:|---:|
| Original main inference | 627 | 12,365,632 |
| Deduplicated QA judgments | 16 | 6,200 |
| Development/regression | 21 | 250,457 |
| Additional 502-page confirmation | 62 | 1,306,292 |
| **Total** | **726** | **13,928,581** |

Prompt 13,788,100 plus completion 140,481. All usage is known and budgets pass. Prior physical context calibration is not billed again. Structure normalization, slicing, deterministic addition and replay make no new generation calls. Private tariff is unknown, so no currency amount is invented.

**32 related unit tests and 51 private trajectory replays pass.** The public verifier independently checks receipt arithmetic, ceilings, source fingerprints, duplicates and per-page aggregate arithmetic; it does not certify the private replay receipts or semantic ground truth. Three frozen inference modules are published byte-for-byte. Public v2.1 tests change only the import of the portable scalar-recovery helper; the assertions remain identical.

```bash
python3 -S scripts/verify_visual_collective_results.py
python3 -S -m unittest tests.test_multimodal_page_harness tests.test_multimodal_page_harness_v2 tests.test_visual_count_memory_public
```

No raw images, labels, prompts, replies, credentials or developer paths are published. All orchestration and outputs were local; the existing Qwen inference service was used without remote file changes. This release adds an opt-in capability, with no replacement of existing defaults or historical scores. Additive census results do not establish arbitrary cross-page joint reasoning or a larger model attention window.
