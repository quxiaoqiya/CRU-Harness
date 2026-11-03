"""Run the current structured-fact Harness without a model or network.

From the repository root::

    python examples/demo_harness.py

The planner, classifier, and character-based token counts are SCRIPTED.
This demonstrates execution and source binding, not model quality, measured
token savings, or benchmark performance. The labels are explicit synthetic
markers so that this example does not pretend to perform semantic inference.
Only the Python standard library is required.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys


# Keep the direct script command usable from a clean checkout, without an
# editable install or a user-specific PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from long_context_agent import AdaptiveV3Config, LongContextHarness
from long_context_agent.experiment_budget import BudgetedClient, ExperimentBudget


class ScriptedTransport:
    """An in-process fake; no HTTP request or real tokenizer is involved."""

    model = "offline-scripted-demo"

    def count(self, *, messages=None, text=None, **_kwargs):
        """Approximate lengths solely to exercise the budget controller."""
        if messages is None:
            return max(1, (len(text or "") + 3) // 4)
        return 5 + sum(
            4 + max(1, (len(message["content"]) + 3) // 4)
            for message in messages
        )

    def request(self, path, body):
        if path != "/chat/completions":
            raise ValueError("The offline transport only simulates completions")
        messages = body["messages"]
        instruction = messages[0]["content"]
        if "Plan evidence acquisition" in instruction:
            reply = {
                "kind": "records",
                "operation": "count",
                "filters": {},
                "field": "label",
                "labels": ["positive", "negative"],
                "target": "positive",
                "answer_template": "Answer: {value}",
            }
        elif "Classify or extract" in instruction:
            rows = json.loads(messages[-1]["content"].split("\nRECORDS:\n", 1)[1])
            classified = []
            for row in rows:
                if "semantic=positive;" in row["text"]:
                    label = 0
                elif "semantic=negative;" in row["text"]:
                    label = 1
                else:
                    raise ValueError("Missing synthetic label marker")
                classified.append([row["id"], label])
            reply = {"records": classified}
        else:
            raise ValueError("Unexpected prompt in the offline demonstration")

        content = json.dumps(reply)
        return {
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": self.count(messages=messages),
                "completion_tokens": self.count(text=content),
            },
        }


def main() -> int:
    context = "The following lines contain 300 records, one per line.\n" + "".join(
        "Date: 2024-01-01 || User: 7 || Instance: "
        f"semantic={'positive' if index < 200 else 'negative'}; shared payload\n"
        for index in range(300)
    )
    budget = ExperimentBudget(
        context_tokens=4_096,
        total_tokens=16_384,
        max_calls=8,
        max_output_tokens=1_024,
    )
    client = BudgetedClient(ScriptedTransport(), budget)
    harness = LongContextHarness(
        None,
        max_context_tokens=budget.context_tokens,
        reserve_tokens=budget.max_output_tokens,
    )
    result = harness.answer_with_structured_facts(
        {"id": "offline-demo", "context": context, "question": "Count positive records."},
        client,
        config=AdaptiveV3Config(
            planner_output_tokens=512,
            classification_output_tokens=1_024,
            final_output_tokens=256,
        ),
    )
    coverage = result.get("semantic_coverage", {})
    if (
        result["mode"] != "adaptive_fact_aggregation_v3"
        or result["answer"] != "Answer: 200"
        or not result["evidence_complete"]
        or coverage.get("unique_projectable_bodies") != 2
        or coverage.get("transferred_records") != 298
        or result["final_fact_loss_count"] != 0
    ):
        raise RuntimeError("Offline demonstration did not produce the expected execution")

    # Display one representative source reference without printing the source
    # material or the full ledger.
    record_id, facts = next(iter(result["record_facts"].items()))
    evidence = facts["label"]["evidence"][0]
    output = {
        "demo": "SCRIPTED planner/classifier and token counts; NOT a benchmark",
        "mode": result["mode"],
        "answer": result["answer"],
        "evidence_complete": result["evidence_complete"],
        "semantic_labels_independently_verified": result["semantic_labels_independently_verified"],
        "source_records": coverage["source_records"],
        "unique_bodies": coverage["unique_projectable_bodies"],
        "directly_classified_records": coverage["direct_model_classified_records"],
        "transferred_records": coverage["transferred_records"],
        "known_records": coverage["known_physical_records"],
        "unknown_records": coverage["unknown_physical_records"],
        "final_fact_loss_count": result["final_fact_loss_count"],
        "simulated_generation_requests": client.summary()["physical_requests"],
        "source_sha256": result["source_sha256"],
        "fact_digest": result["fact_state"]["fact_digest"],
        "source_binding_example": {
            "record_id": record_id,
            "source_span": evidence["source_span"],
            "source_sha256": evidence["source_sha256"],
            "stage": evidence["provenance"]["stage"],
        },
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
