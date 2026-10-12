# DECISIONS

One line per decision not settled by PLAN.md, with the reason.

- 2026-10-12: Escalations use `fable-debugger` (availability check returned OK), `opus-debugger` only as fallback.
- 2026-10-12: Cloned menagerie with plain `git clone` + checkout of Playground's pinned SHA (identical to `ensure_menagerie_exists()`), in the background in parallel with the pip install, to save time.
- 2026-10-12: Probe scripts that are not part of the pipeline live in `scripts/dev/` (kept for transparency).
