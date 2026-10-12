"""Physics sanity: a hand-written controller (true hole position) can insert."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
import scripted_insert  # noqa: E402

N = 20


@pytest.mark.parametrize('clearance_mm,min_rate', [(2.0, 0.95), (1.0, 0.90)])
def test_scripted_centred_insertion(clearance_mm, min_rate):
  env = scripted_insert.make_env(clearance_mm)
  r = scripted_insert.run_scripted(env, n=N, seed=1)
  assert r['success_rate'] >= min_rate, r


def test_scripted_half_mm_recorded():
  env = scripted_insert.make_env(0.5)
  r = scripted_insert.run_scripted(env, n=N, seed=1)
  print(f"scripted 0.5 mm centred: {r}")
  assert 0.0 <= r['success_rate'] <= 1.0
