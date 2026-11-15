# Take-home decisions

- 2026-11-15: Routine helpers use gpt-6-luna with low reasoning, explicitly selected on each spawn from takehome/routine_helper.toml; runtime session records verified the requested model and effort.
- 2026-11-15: Preflight used eight test-only episodes at seed 123; none were learned from, so practice usage and experimental settings tried remain zero.
- 2026-11-15: Baseline evaluation retains the existing evaluator's batch size 256, contact capacity 20 * 256, domain randomization, and deterministic policy to reproduce the recorded numbers without changing the test.
- 2026-11-15: The no-camera-error comparison removes the constant sideways bias only and retains step-to-step noise, matching the existing 94.0% reference; reports will call this the no-camera-bias ceiling precisely.
- 2026-11-15: Python runs use -B and redirect generated caches under results/takehome/; reuse the preflight cache to avoid repeating compilation and preserve existing project files.

Settings tried so far: 0 learning settings; evaluation and preflight checks are not learning experiments.

- 2026-11-15: Initial aim correction uses a maximum 0.1 mm nudge per axis per step and a 3 mm running-guess cap per axis; gradual corrections can cover four standard deviations of the specified bias.
- 2026-11-15: Implement the wrapper as a subclass in takehome/, because the existing domain-randomization wrapper calls the unwrapped environment directly and would bypass ordinary composition.
- 2026-11-15: Keep physics running for fixed-length evaluation while recording a sticky learner terminal flag and excluding subsequent transitions from learning; preserve the original test's full-episode measurement and end the learner at first success.
- 2026-11-15: Reserve and persist practice episodes before collection, counting unfinished reservations conservatively; process interruption must not free previously allocated practice budget.
- 2026-11-15: Do not resolve the conflicting universal 50/50 sampling requirement and empty-buffer experiments without Griffin's answer, as AGENTS.md rule 8 explicitly requires asking about an impossible plan requirement.
- 2026-11-15: The previous sampling pause is resolved by Griffin and the revised instructions: exactly 50/50 when both buffers contain data, otherwise sample the nonempty source; follow only the revised When to stop rule going forward.
- 2026-11-15: Keep the old requirements extraction as historical evidence and write goal4_requirements_revised.json for the smaller current scope; no later-goal tricks will be built during Goal 4.
