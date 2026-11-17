# Rebuilding and resuming the follow-up

Use WSL, the `takehome` branch, and the existing local project directory. Every shell starts with `source scripts/env.sh`. Data outputs and caches stay under `results/takehome/followup/`; the report assembler writes the permitted top-level `TAKEHOME_FOLLOWUP_REPORT.md`. These commands do not rewrite the original study.

## Rebuild the small report artifacts

These commands use saved evidence and do not run practice or test episodes:

```bash
source scripts/env.sh
python -B -m takehome.followup_plot_a
python -B -m takehome.followup_offset_plot
python -B -m takehome.followup_mixed_report
python -B -m takehome.followup_collect
python -B -m takehome.followup_scoreboard --plot-phase B
python -B -m takehome.followup_scoreboard --plot-phase B_mixed
python -B -m takehome.followup_scoreboard --plot-phase C
python -B -m takehome.followup_scoreboard --plot-phase D
python -B -m takehome.followup_scoreboard --plot-phase E
python -B -m takehome.followup_accounting
python -B -m takehome.followup_report
```

The lead-written phase sections are in `results/takehome/followup/report_sections/`. Report assembly includes the whole scoreboard, including skips and the preserved partial run. The chart whiskers show the sample standard deviation, not a confidence interval. The Phase A plot reads `A/diagnosis.json`; the offset plot requires all 49 saved `B/offset_sweep/cell_*/test_0000.json` files.

Do not run the collector or another suite while a suite parent is active: the parent is the single writer for the registry and scoreboard. The report can be rebuilt after a phase finishes. Large traces and models are local ignored files; regenerating metrics from traces requires those local files. Committed JSON summaries and charts remain available without them.

After Phase G has completed, rebuild its confirmation table and chart with:

```bash
source scripts/env.sh
python -B -m takehome.followup_final_report
```

This report command reads the written choice and completed results; it never starts a fresh run.

## Resume an interrupted development queue

Read `TAKEHOME_PROGRESS.md`, `registry.json`, the suite log, and its `.exit.json` first. A live parent must finish or be diagnosed before another parent starts. A tripwire halt requires the stopping-rule process; do not clear it to resume a command.

The registry supplies each setting's configuration, base, declared seeds and output path. A suite runs at most four settings at once, with each setting's seeds batched in one process. Use a new log name when resuming because logs are created exclusively. For example, if these settings are already authorized and incomplete:

```bash
source scripts/env.sh
python -B -m takehome.followup_detach --name c_resume_1 -- \
  python -B -m takehome.followup_suite --name c_resume_1 \
  --settings C_fill_0 C_fill_10 C_fill_20 C_fill_80
```

Completed `result.json` files are skipped. A partial experiment resumes its saved checkpoint and reserved practice data; it refuses to recollect a charged batch with no saved data. Each experiment retains its own 200-episode ledger. Only a completed three-seed PROMISING result is extended to the predeclared five seeds. For a condition whose base needs additional matched seeds, measure those before comparing the extension.

The historical halted fixed-camera seed 4 is intentionally not resumed. Its separately authorized fresh run is reached through `seed_paths` in the registry, preserving both directories.

## Verification

The isolated harness sends all test scratch files and caches to the follow-up directory and records any simulated test or reset probes:

```bash
source scripts/env.sh
python -B -m takehome.followup_tests tests takehome
```

After the one permitted pointer paragraph is added to the original report in Phase G, verify the original artifacts with:

```bash
source scripts/env.sh
python -B -m takehome.followup_integrity verify --allow-report-append
```

Before that paragraph is added, omit `--allow-report-append`. Use the retained `preservation_manifest.json` from before the follow-up; do not refresh it against changed files. The manifest checks hashes and timestamps for the original source, policies and results, including large files on disk.

## Locked final confirmation

After F is complete and the lead has written G/choice.json, measure the fresh reference once, then use the supervised four-worker command. The supervisor checks completed methods for identical per-episode outcomes and stops siblings if a tripwire fires. It refuses to start before the preceding phases finish.

```bash
source scripts/env.sh
python -B -m takehome.followup_detach --name g_references -- \
  python -B -m takehome.followup_final_reference
```

After the reference command's exit file reports success:

```bash
source scripts/env.sh
python -B -m takehome.followup_detach --name g_confirmation -- \
  python -B -m takehome.followup_final_suite --name g_confirmation
```

Use a new log name for an authorized resume. The written choice and existing final results must stay unchanged.
