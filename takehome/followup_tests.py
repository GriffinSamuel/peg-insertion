"""Run tests with all new caches and scratch files inside followup."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from takehome.followup_runtime import OUT
import pytest
if __name__=='__main__':
    targets=sys.argv[1:] or ['tests','takehome']
    from takehome.followup_test_audit import TestAudit
    audit=TestAudit();audit.install()
    code=pytest.main(['-q','-p','no:cacheprovider','--basetemp',
        str(OUT/'cache/pytest'),*targets],plugins=[audit])
    audit.save(code)
    raise SystemExit(code)
