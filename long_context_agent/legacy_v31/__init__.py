"""Unmodified v3.1 controller from the sealed cohort4 experiment.

Relative imports keep the complete v3.1 ledger/parser/protocol isolated from
subsequent implementation changes. Only the shared budgeted client crosses
this boundary. See manifest.json for the original source hashes.
"""
from .adaptive_evidence_v3 import AdaptiveV3Config, run_adaptive_v3
__all__ = ["AdaptiveV3Config", "run_adaptive_v3"]
