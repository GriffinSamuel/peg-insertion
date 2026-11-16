"""Run original and takehome tests with allowed cache paths set before imports."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from takehome import runtime  # noqa: E402,F401
import pytest  # noqa: E402


if __name__ == '__main__':
    requested = sys.argv[1:] or ['tests', 'takehome', '-q']
    raise SystemExit(pytest.main([
        *requested,
        '-o', 'cache_dir=results/takehome/pytest_cache',
        '--basetemp=results/takehome/final_test_tmp',
    ]))
