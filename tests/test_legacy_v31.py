"""The old controller must stay isolated from mutable v3.2 source files."""
from pathlib import Path
import hashlib
import json

from long_context_agent import adaptive_evidence_v3 as current
from long_context_agent.legacy_v31 import adaptive_evidence_v3 as old
from long_context_agent.legacy_v31 import fact_ledger as old_ledger
from scripts.benchmark_context_holdout import source_fingerprints


def test_snapshot_matches_manifest_and_optional_local_archive():
    root = Path(old.__file__).parent
    manifest = json.loads((root / 'manifest.json').read_text())
    project = root.parents[1]
    original = project / manifest['source_experiment'] / 'frozen_source/long_context_agent'
    for name, digest in manifest['files'].items():
        data = (root / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest
        # The portable release carries the immutable package and its digest
        # manifest. If a full local experiment archive is present, check it too.
        if original.is_dir():
            assert data == (original / name).read_bytes()


def test_mutable_ledger_and_config_are_not_shared():
    assert old.FactLedger is old_ledger.FactLedger
    assert old.FactLedger is not current.FactLedger
    assert old.AdaptiveV3Config is not current.AdaptiveV3Config
    fingerprints = source_fingerprints()
    assert fingerprints['long_context_agent/legacy_v31/adaptive_evidence_v3.py']
    assert fingerprints['long_context_agent/legacy_v31/manifest.json']
