"""Watch full inference, then score/audit/export locally without extra user steps."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

from scripts import benchmark_longmemeval_v2_full as f
from scripts.audits.export_longmemeval_v2_full import live,export

b=f.b


def main(provider_env=None):
    work=f.WORK;out=work/'formal';status=work/'completion_pipeline.json'
    with (work/'.completion.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        def mark(phase,**kwargs):
            b.atomic(status,{'phase':phase,'updated_at_unix':time.time(),**kwargs})
            print(json.dumps({'phase':phase,**kwargs}),flush=True)
        mark('waiting_for_all_931_physical_jobs')
        deadline=time.monotonic()+24*3600;unlocked_checks=0
        while True:
            live(work)
            if (out/'run_summary.json').exists():
                summary=json.loads((out/'run_summary.json').read_text())
                if summary['phase']=='inference_complete':break
                mark('needs_attention_inference_incomplete',summary=summary)
                live(work,'needs_attention_inference_incomplete');return
            with (out/'.run.lock').open('a+') as probe:
                try:
                    fcntl.flock(probe.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
                    unlocked_checks+=1
                except BlockingIOError:unlocked_checks=0
            if unlocked_checks>=3:
                mark('needs_attention_worker_stopped',reason='No inference lock and no terminal summary')
                return
            if time.monotonic()>deadline:
                mark('needs_attention_wait_timeout');return
            time.sleep(30)
        provider=b.load_provider_env(provider_env)
        if provider['model']!=b.MODEL:raise ValueError('Model identity mismatch')
        mark('scoring_all_fixed_predictions');live(work,'scoring_all_fixed_predictions')
        f.score(work,provider)
        mark('offline_replay_all_physical_records');live(work,'offline_replay_all_physical_records')
        subprocess.run([sys.executable,'-B','-m','scripts.audits.audit_longmemeval_v2_full',str(work)],cwd=f.ROOT,check=True)
        mark('exporting_final_report');export(work)
        subprocess.run(['/usr/bin/python3','-S','scripts/verify_longmemeval_v2_full_results.py'],cwd=f.ROOT,check=True)
        mark('complete',report=str(f.ROOT/'docs/longmemeval_v2_full_20261104.zh-CN.md'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--provider-env',type=Path,help='Local provider configuration; alternatively set Q38_API_KEY')
    args=parser.parse_args()
    try:main(args.provider_env)
    except Exception as exc:
        b.atomic(f.WORK/'completion_pipeline_error.json',{'error_type':type(exc).__name__,'at_unix':time.time()})
        raise
