"""Reuse the pinned replay checks for the separately frozen conservative route."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts import benchmark_longmemeval_v2_visual_route as route
from scripts.audits import audit_longmemeval_v2_visual as audit
if __name__=='__main__':
    audit.WORK=route.WORK
    audit.main()
