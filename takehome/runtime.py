"""Keep generated caches within the allowed output directory; import first."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _key, _sub in {
    "XDG_CACHE_HOME": "xdg", "JAX_COMPILATION_CACHE_DIR": "jax",
    "WARP_CACHE_PATH": "warp", "MPLCONFIGDIR": "mpl",
}.items():
    os.environ[_key] = str(ROOT / "results/takehome/preflight_cache" / _sub)
