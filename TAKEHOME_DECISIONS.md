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
- PLAN GAP 2026-11-15: Use a deterministic off-policy actor-critic, with four layer-normalized critics and the minimum of two randomly selected target critics; the plan specifies the ingredients but not the algorithm details, and this is a small implementation of that recipe rather than an exact SAC/REDQ reproduction.
- PLAN GAP 2026-11-15: Use two 64-unit hidden layers, Adam at 0.0003, batch size 128, four critic updates per valid new learner transition, and one actor update per four critic updates; choose a modest main configuration before seeing any learning result.
- PLAN GAP 2026-11-15: Use discount 0.99 and target-network averaging rate 0.005; stop the backup on true termination but retain it at a time limit, matching the revised definition that raw Q is not a success probability.
- PLAN GAP 2026-11-15: Fill-up nudges use independent Gaussian noise with normalized standard deviation 0.5, learner exploration uses 0.2, and both are clipped to [-1,1]; with a 0.1 mm step cap this gives small exploration while preserving zero network output at initialization.
- PLAN GAP 2026-11-15: Store proposed corrections in metres and divide by their caps at the critic input; scale the running guess by its 3 mm cap, and normalize the original sensor vector with the frozen checkpoint's exact statistics.
- PLAN GAP 2026-11-15: Run the ten-episode code trial at practice seed 999 with zero fill-up and eight test-only episodes at seed 123; a single ten-episode batch can exercise the learner while unit tests separately verify mixed-source sampling, without touching confirmation seeds.
- PLAN GAP 2026-11-15: Include a 256-test-episode point at the end of fill-up (40), then at 60 through 180, and 500 at 200; point zero reuses the original-policy baseline, and all evaluations use test seed 123.
- PLAN GAP 2026-11-15: Output dimensions 2/3/5 are supported and zero-initialization-tested in the network; movement environment adapters and all other Goal 6/7 features are deferred as explicitly required by the reduced Goal 4 scope.
- PLAN GAP 2026-11-15: The required 1/2/4-process and multi-seed-process speed tests need more than five runs; predeclare nine independent 200-episode runs of one setting, report every run, and use only the first designated run for each practice seed in the main result, with four repeats used solely for timing and reproducibility.
- PLAN GAP 2026-11-15: Benchmark schedule is one process/seed 0; two concurrent processes/seeds 1 and 2; four concurrent processes/seeds 3, 4, 0 and 1; then one process running seeds 0 and 1 sequentially. The first appearances of 0–4 are the headline runs; repeats never replace them based on success.
