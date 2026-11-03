"""Local fake-data checks for holdout pairing, sealed labels and resume quotas."""
from __future__ import annotations

from argparse import Namespace
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from long_context_agent.experiment_budget import ExperimentBudget
from scripts import analyze_context_holdout as scoring
from scripts import benchmark_context_holdout as runner


class NoModelTransport:
    model = "fake-model"

    def __init__(self):
        self.requests = 0

    def count(self, **kwargs):
        return 10

    def request(self, *args, **kwargs):
        self.requests += 1
        raise AssertionError("A completed or interrupted task must not issue another generation")


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _fixture(base):
    """Fabricated records only; never references the real held-out directory."""
    base = Path(base)
    output = base / "out"
    output.mkdir()
    inputs = base / "inputs.jsonl"
    samples = [
        {"id": "sample-z", "context": "fabricated source z", "question": "fabricated question", "metadata": {"context_len": 32768}},
        {"id": "sample-a", "context": "fabricated source a", "question": "fabricated question", "metadata": {"context_len": 65536}},
    ]
    inputs.write_text("".join(json.dumps(s) + "\n" for s in samples))
    labels = base / "labels.json"
    _write_json(labels, {"sample-z": {"answer": "[10]", "answer_type": "ANSWER_TYPE.NUMERIC"}, "sample-a": {"answer": "[20]", "answer_type": "ANSWER_TYPE.NUMERIC"}})
    source = base / "source/eval_helpers_original.py"
    source.parent.mkdir()
    source.write_text("# fabricated scorer reference; not executed\n")
    manifest = base / "manifest.json"
    _write_json(manifest, {"fixture": True})
    frozen = {"methods": ["adaptive", "rlm"], "sample_ids": [s["id"] for s in samples], "data_manifest": str(manifest), "labels_sha256": runner.digest_file(labels), "official_scorer_sha256": runner.digest_file(source), "model": "fake-model", "budget": {"context_tokens": 16384, "total_tokens": 65536, "max_calls": 8, "max_output_tokens": 2048}}
    _write_json(output / "freeze.json", frozen)
    _write_json(output / "context_measurements.json", {"fixture": True})
    return Namespace(inputs=str(inputs), labels=str(labels), output=str(output)), frozen, samples


def _prediction(sample, method, freeze_sha, answer):
    return {"id": sample["id"], "method": method, "freeze_sha256": freeze_sha,
            "context_sha256": hashlib.sha256(sample["context"].encode()).hexdigest(),
            "context_len": sample["metadata"]["context_len"], "task_type": "fake-task",
            "prediction": answer, "error": None, "seconds": 1,
            "result": {"baseline_identity": {"name": "fabricated test method"}},
            "usage": {"physical_requests": 1, "charged_tokens": 20, "prompt_tokens": 18, "completion_tokens": 2, "within_budget": True, "all_usage_known": True}}


def _all_predictions(args, frozen, samples):
    output = Path(args.output)
    freeze_sha = runner.digest_file(output / "freeze.json")
    answers = {("sample-z", "adaptive"): "Answer: 10", ("sample-a", "adaptive"): "Answer: 21", ("sample-z", "rlm"): "Answer: 11", ("sample-a", "rlm"): "Answer: 20"}
    for sample in reversed(samples):
        for method in reversed(frozen["methods"]):
            _write_json(output / f"predictions/{sample['id']}__{method}.json", _prediction(sample, method, freeze_sha, answers[sample["id"], method]))


class HoldoutRunnerTests(unittest.TestCase):
    def test_interrupted_journal_preserves_charges_and_never_replays(self):
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            root = Path(temporary)
            sample = {"id": "interrupted", "context": "fake source", "question": "fake question"}
            reservation = {"physical_request": 1, "admission_prompt_tokens": 20,
                           "requested_output_tokens": 12, "charged_tokens": 32,
                           "status": "inflight"}
            _write_json(root / "journals/interrupted__adaptive.json", {"freeze_sha256": "sealed", "ledger": [reservation]})
            transport = NoModelTransport()
            result = runner.run_one(sample, "adaptive", transport, ExperimentBudget(), root, "sealed")
            self.assertEqual(result["error"]["type"], "InterruptedTask")
            self.assertEqual(result["usage"]["charged_tokens"], 32)
            self.assertEqual(result["usage"]["physical_requests"], 1)
            self.assertEqual(result["prediction"], "")
            self.assertEqual(transport.requests, 0)
            repeated = runner.run_one(sample, "adaptive", transport, ExperimentBudget(), root, "sealed")
            self.assertEqual(repeated, result)
            self.assertEqual(transport.requests, 0)

    def test_other_freeze_predictions_cannot_be_reused(self):
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            sample = {"id": "different-freeze", "context": "x", "question": "q"}
            _write_json(Path(temporary) / "predictions/different-freeze__adaptive.json", {"freeze_sha256": "old"})
            transport = NoModelTransport()
            with self.assertRaisesRegex(ValueError, "another freeze"):
                runner.run_one(sample, "adaptive", transport, ExperimentBudget(), temporary, "new")
            self.assertEqual(transport.requests, 0)

    def test_incomplete_predictions_do_not_read_or_hash_labels(self):
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            args, frozen, samples = _fixture(temporary)
            output = Path(args.output)
            freeze_sha = runner.digest_file(output / "freeze.json")
            _write_json(output / "predictions/sample-z__adaptive.json", _prediction(samples[0], "adaptive", freeze_sha, "Answer: 10"))
            original_text, original_bytes = Path.read_text, Path.read_bytes
            label_path = Path(args.labels)

            def guarded_text(path, *pos, **kw):
                if path == label_path:
                    raise AssertionError("Labels opened before every inference pair exists")
                return original_text(path, *pos, **kw)

            def guarded_bytes(path, *pos, **kw):
                if path == label_path:
                    raise AssertionError("Label hash read before every inference pair exists")
                return original_bytes(path, *pos, **kw)

            with patch.object(scoring, "check_freeze", return_value=frozen), patch.object(Path, "read_text", guarded_text), patch.object(Path, "read_bytes", guarded_bytes):
                with self.assertRaisesRegex(ValueError, "Inference incomplete"):
                    scoring.analyze(args)
            self.assertFalse((output / "scoring_access.json").exists())

    def test_scoring_pairs_by_input_identity_not_filename_or_write_order(self):
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            args, frozen, samples = _fixture(temporary)
            output = Path(args.output)
            # Lexical ID order and creation order both differ from input order.
            _all_predictions(args, frozen, samples)
            with patch.object(scoring, "check_freeze", return_value=frozen), patch.object(scoring, "bootstrap_difference", wraps=scoring.bootstrap_difference) as bootstrap, redirect_stdout(io.StringIO()):
                scoring.analyze(args)
            left, right = bootstrap.call_args.args
            self.assertEqual(left, [1.0, 0.75])
            self.assertEqual(right, [0.75, 1.0])
            audit = json.loads((output / "audit.json").read_text())
            summary = audit["summary"]["adaptive"]
            self.assertEqual(summary["mean_official_score"], 0.875)
            self.assertEqual(summary["full_credit"], 1)
            self.assertEqual(summary["exact_match"], 1)
            self.assertEqual(summary["responded"], 2)
            comparison = audit["paired_comparisons"]["adaptive_minus_rlm"]
            self.assertEqual(comparison["mean_difference"], 0)
            self.assertEqual(comparison["paired_bootstrap_95_percentile_ci"], [-0.25, 0.25])

    def test_swapped_pair_records_fail_before_reading_labels(self):
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            args, frozen, samples = _fixture(temporary)
            _all_predictions(args, frozen, samples)
            path = Path(args.output) / "predictions/sample-z__rlm.json"
            record = json.loads(path.read_text())
            record["method"] = "adaptive"
            _write_json(path, record)
            with patch.object(scoring, "check_freeze", return_value=frozen):
                with self.assertRaisesRegex(ValueError, "identity or source mismatch"):
                    scoring.analyze(args)
            self.assertFalse((Path(args.output) / "scoring_access.json").exists())

    def test_v3_scoring_focus_compares_old_controller_on_identical_pairs(self):
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            args, frozen, samples = _fixture(temporary)
            frozen['methods'] = ['adaptive_v3', 'adaptive', 'rlm']
            output = Path(args.output)
            _write_json(output / 'freeze.json', frozen)
            freeze_sha = runner.digest_file(output / 'freeze.json')
            for sample in samples:
                for method in frozen['methods']:
                    answer = 'Answer: 10' if sample['id'] == 'sample-z' else 'Answer: 20'
                    record = _prediction(sample, method, freeze_sha, answer)
                    if method == 'adaptive_v3':
                        record['result'].update(mode='adaptive_fact_aggregation_v3', record_facts=[{'value': 'x'}],
                            aggregate={'known_count': 1, 'unknown_count': 2}, answer_is_estimate=True,
                            final_fact_loss_count=0, trace=[{'action': 'deterministic_finalize'}])
                    _write_json(output / f"predictions/{sample['id']}__{method}.json", record)
            with patch.object(scoring, 'check_freeze', return_value=frozen), redirect_stdout(io.StringIO()):
                scoring.analyze(args)
            report = json.loads((output / 'audit.json').read_text())
            self.assertEqual(report['focus_method'], 'adaptive_v3')
            self.assertEqual(set(report['paired_comparisons']), {'adaptive_v3_minus_adaptive', 'adaptive_v3_minus_rlm'})
            self.assertEqual(len(report['focus_mechanisms']), 2)
            self.assertEqual(report['focus_mechanisms'][0]['unknown_count'], 2)
            self.assertEqual(report['focus_mechanisms'][0]['final_fact_loss_count'], 0)
            self.assertTrue(report['focus_mechanisms'][0]['deterministic_final'])
            self.assertIn('2题小样本', (output / 'audit.md').read_text())

    def test_rescoring_preserves_first_label_access_timestamp(self):
        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            args, frozen, samples = _fixture(temporary)
            _all_predictions(args, frozen, samples)
            access = Path(args.output) / "scoring_access.json"
            _write_json(access, {"labels_first_scored_at_unix": 1234})
            with patch.object(scoring, "check_freeze", return_value=frozen), redirect_stdout(io.StringIO()):
                scoring.analyze(args)
            audit = json.loads(access.read_text())
            self.assertEqual(audit["labels_first_scored_at_unix"], 1234)
            self.assertGreater(audit["labels_last_scored_at_unix"], 1234)

    def test_freeze_and_run_skeleton_measures_context_without_opening_labels(self):
        class FakeInferenceTransport:
            model = "fake-model"

            def __init__(self):
                self.generations = 0
                self.model_lists = 0

            def count(self, **kwargs):
                return 10

            def request(self, path, body=None):
                if path == "/models":
                    self.model_lists += 1
                    return {"data": [{"id": self.model, "max_model_len": 4096}]}
                if path == "/chat/completions":
                    self.generations += 1
                    return {"choices": [{"message": {"content": "10"}}], "usage": {"prompt_tokens": 10, "completion_tokens": 1}}
                raise AssertionError("Unexpected request " + path)

        with tempfile.TemporaryDirectory(dir="/private/tmp") as temporary:
            args, _, samples = _fixture(temporary)
            output = Path(args.output)
            (output / "freeze.json").unlink()
            (output / "context_measurements.json").unlink()
            manifest_path = Path(args.inputs).parent / "manifest.json"
            _write_json(manifest_path, {"inputs_sha256": runner.digest_file(args.inputs), "labels_sha256": runner.digest_file(args.labels), "scorer_sha256": runner.digest_file(Path(args.inputs).parent / "source/eval_helpers_original.py")})
            args.methods, args.model = "adaptive", "fake-model"
            args.context_tokens, args.total_tokens = 512, 2048
            args.max_calls, args.max_output_tokens = 3, 64
            args.provider_env, args.timeout, args.workers = None, 3, 2
            transport = FakeInferenceTransport()
            original_text, original_bytes = Path.read_text, Path.read_bytes
            label_path = Path(args.labels)

            def guard_text(path, *pos, **kw):
                if path == label_path:
                    raise AssertionError("Freeze/run opened the sealed fake labels")
                return original_text(path, *pos, **kw)

            def guard_bytes(path, *pos, **kw):
                if path == label_path:
                    raise AssertionError("Freeze/run hashed the sealed fake labels")
                return original_bytes(path, *pos, **kw)

            (Path(temporary) / "fake.py").write_text("# fabricated source snapshot\n")
            with patch.object(runner, "ROOT", Path(temporary)), patch.object(runner, "source_fingerprints", return_value={"fake.py": "hash"}), patch.object(runner, "runtime_identity", return_value={"runtime": "fake"}), patch.object(runner, "load_provider_env", return_value={"base_url": "http://unused.invalid", "api_key": "fake", "model": "unused"}), patch.object(runner, "ProviderTransport", return_value=transport), patch.object(Path, "read_text", guard_text), patch.object(Path, "read_bytes", guard_bytes), redirect_stdout(io.StringIO()):
                runner.freeze(args)
                runner.run(args)
                # A second invocation must reuse complete pair files.
                runner.run(args)
            frozen = json.loads((output / "freeze.json").read_text())
            measured = json.loads((output / "context_measurements.json").read_text())
            self.assertEqual(frozen["sample_ids"], [s["id"] for s in samples])
            self.assertEqual(len(measured["samples"]), 2)
            self.assertEqual(transport.model_lists, 1)
            self.assertEqual(transport.generations, 2)
            self.assertEqual((output / "frozen_source/fake.py").read_text(), "# fabricated source snapshot\n")
            self.assertTrue(all(row["provider_reported_native_limit"] == 4096 for row in measured["samples"]))

    def test_paired_bootstrap_retains_the_difference_in_each_pair(self):
        result = scoring.bootstrap_difference([0.7, 0.9], [0.2, 0.4], repetitions=1000)
        self.assertAlmostEqual(result["mean_difference"], 0.5)
        self.assertEqual(result["n_pairs"], 2)
        for bound in result["paired_bootstrap_95_percentile_ci"]:
            self.assertAlmostEqual(bound, 0.5)


if __name__ == "__main__":
    unittest.main()
