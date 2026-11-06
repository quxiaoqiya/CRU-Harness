# Optional visual page memory

These modules are opt-in; they do not change the existing core or trajectory
defaults. Page-memory v2/v2.1 is separate from trajectory visual V3 and core V3.2.
See the [evaluation](visual_collective_20261105.md) and
[中文报告](visual_collective_20261105.zh-CN.md) for measured gains and failures.

## Original pixels and source identity

`PageIndex` accepts original rendered PNG records:

```python
import hashlib
from pathlib import Path
from long_context_agent.multimodal_page_harness import PageIndex

root = Path("local_pages")
rows = []
for number, path in enumerate(sorted(root.glob("*.png")), start=1):
    rows.append({
        "id": f"p{number:04d}", "doc_id": "all-pages.bundle", "page": number,
        "text": "", "image": path.name,
        "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    })
index = PageIndex(rows, root)
```

Empty text means no supplied digital-text channel. It does not mean the image
contains no objects or that an OCR model has already read it. Image bytes are
checked before use; images count as exposed only after a generation using them
has completed. Do not put answers or reference counts into index records.

## Source-bound additive counts

Configure credentials through local environment variables `Q38_API_KEY`,
`Q38_BASE_URL`, and `Q38_MODEL`; credentials are not included in published data.
The existing `ProviderTransport` expects an OpenAI-compatible chat endpoint with
native tools and a serving `/tokenize` endpoint. It is not a universal provider
adapter. Use a separate metered client per method and task.

```python
from long_context_agent.experiment_budget import (
    BudgetedClient, ExperimentBudget, ProviderTransport, load_provider_env,
)
from long_context_agent.visual_count_memory_v21 import (
    CountMemoryConfig, run_count_memory,
)

client = BudgetedClient(
    ProviderTransport(**load_provider_env()),
    ExperimentBudget(
        context_tokens=262144, total_tokens=3145728, max_calls=160,
        max_output_tokens=16384, seed=20261105,
    ),
)
result = run_count_memory(
    "Count every Picture block and every Table block across all supplied pages.",
    index, [row["id"] for row in rows], client,
    config=CountMemoryConfig(batch_pages=16, repair_pages=8),
    # on_checkpoint=persist_private_fact_state,
)
print(result["answer"], client.summary())
```

Each source has one integer vector. The count prompt defines block counting,
including logos and grouped multi-panel figures, rather than every semantic
object. Accepted facts are deduplicated by source ID; disagreements become
conflicts, and unresolved or missing pages prevent a complete total. Only
missing/conflicting sources are reread, at most three observations per page and
64 repair calls, within global admission limits. Finished pixel-cache entries
are evicted; structured facts remain. Checkpoints expose facts and traces for
durable storage; automatic process-resume or request-resend is not implemented.

An anonymous list is position-bound only at exactly the expected length. An
explicit ID can bind a partial response without shifting another page's count.
Malformed, negative, non-finite, boolean or unknown values do not become zero.
A transport exception propagates: an uncertain request is not automatically
resent as a count-format repair.

`CountFacts` prevents double-counting repeated observations of one source ID.
It does **not** merge different page IDs merely because their pixels match;
whether repeated pages count twice depends on task semantics. For a new holdout,
exclude previous source names **and pixel SHA256**, and check duplicates within
the new cohort before freezing. The released result includes one disclosed
cross-filename duplicate and a post-hoc slice; it does not conceal that mistake.

Distinct-person or unique-entity counting cannot use integer addition. This API
rejects `distinct`/`unique` requests; it does not implement cross-page identity
resolution. Full fact coverage also cannot guarantee correct visual counts.

## Coverage retrieval and final recovery

```python
from long_context_agent.multimodal_page_harness_v2 import run_memory_v2

# Use a fresh independent client for a different task/method.
answer = run_memory_v2(question, index, client)
```

This v2 QA route augments text search with low-text-page coverage, advances past
already returned pages, and reserves a plain-chat final when tool output is
empty or repeatedly provides no fresh pixels. The final call attaches actual
observed images. Its legacy automatic count routing still uses frozen v2;
call `run_count_memory` explicitly for the v2.1 additive-count protocol.
Do not use that automatic route for cross-page distinct entities.

`visual_count_outputs.explicit_global_answer` is a portable extraction helper
copied from the local audit finalizer. Call it only when **all** required pages
were actually supplied in one request. It never recovers unseen facts or
promotes a nested page-level answer into a global count. Its publication does
not change the frozen inference modules.

## Offline verification and test lineage

```bash
python3 -S scripts/verify_visual_collective_results.py
python3 -S -m unittest tests.test_multimodal_page_harness tests.test_multimodal_page_harness_v2 tests.test_visual_count_memory_public
```

The three inference modules are published byte-for-byte from the frozen run.
The first two test files are unchanged. The 13-test public v2.1 file changes
only one import: its scalar-recovery test uses the portable helper instead of
the private experiment finalizer. All 32 tests execute without
downloaded data, model calls, or site-packages. The public verifier checks
arithmetic, budgets, receipt totals, source hashes and the duplicate correction;
it does not repeat the 51 private trajectory replays or validate semantic truth.
