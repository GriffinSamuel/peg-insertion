# Archived (superseded) results

These are earlier runs made before the final training configuration was fixed.
They are kept so the history of the project is visible; they are NOT part of the
reported results. See DECISIONS.md (2026-10-12) for why each change was made.

- `lr5e-4_10M/`: first runs with Playground's default learning rate 5e-4 and 10M steps, plus three single-knob diagnostics (diagA/B/C) of the force@0.5mm collapse.
- `lr2e-4_align20mm/`: learning rate 2e-4, 20M steps, original 20 mm-wide alignment reward (pos_1mm_s0 collapsed to 0%).
- `reward_fix_check/`: seed-0 check of the final reward (5 mm alignment term) on four conditions, trained in parallel.

Trained parameters and videos of these runs were moved out of the repository to keep it small.
