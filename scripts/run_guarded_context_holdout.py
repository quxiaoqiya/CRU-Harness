#!/usr/bin/env python3
"""Run a frozen local-Mac inference job with OS-level label read denial."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

from scripts.benchmark_context_holdout import atomic_json, digest_file, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--provider-env')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--timeout', type=float, default=180)
    parser.add_argument('--inside-guard', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    out = Path(args.output)
    profile = out / 'labels_guard.sb'
    frozen = json.loads((out / 'freeze.json').read_text())
    if not args.inside_guard:
        if sys.platform != 'darwin':
            raise RuntimeError('This launcher requires the local Mac sandbox-exec label guard')
        os.execv('/usr/bin/sandbox-exec', ['sandbox-exec', '-f', str(profile), sys.executable,
                                         '-m', 'scripts.run_guarded_context_holdout',
                                         *sys.argv[1:], '--inside-guard'])
    try:
        with open(frozen['label_guard_probe']) as probe:
            probe.read()
    except PermissionError:
        atomic_json(out / 'guard_verification.json', {
            'verified': True, 'probe_read_denied': True, 'labels_not_opened': True,
            'profile_sha256': digest_file(profile), 'time': time.time(),
            'launcher': 'scripts/run_guarded_context_holdout.py'})
    else:
        raise RuntimeError('OS label-read guard is not active; refusing inference')
    print('macOS label-read denial verified; starting frozen inference.', flush=True)
    run(args)


if __name__ == '__main__':
    main()
