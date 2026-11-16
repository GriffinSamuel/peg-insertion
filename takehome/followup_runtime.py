"""Import first: isolate every follow-up cache from the completed study."""
import os
from pathlib import Path
import sys

if 'jax' in sys.modules:
    raise RuntimeError('Import followup_runtime before JAX or experiment modules')

from takehome import runtime as _original_runtime  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/takehome/followup'
for key, sub in {
    'XDG_CACHE_HOME': 'xdg', 'JAX_COMPILATION_CACHE_DIR': 'jax',
    'WARP_CACHE_PATH': 'warp', 'MPLCONFIGDIR': 'mpl',
}.items():
    os.environ[key] = str(OUT / 'cache' / sub)

