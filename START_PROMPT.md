Paste everything below the line as your first message in Claude Code (after `/model` shows Opus as the main model). If Fable was NOT listed under `/model`, add this sentence to the end: "Fable is not available on this plan; run every escalation with the opus model instead and note it in PROGRESS.md."

---

You are starting the peg-insertion project. Read CLAUDE.md completely, then PLAN.md completely, before doing anything else. CLAUDE.md sets the rules (stay inside this folder, no sudo, no pausing except for the listed big-issue conditions, Fable escalation triggers, honest results). PLAN.md is the phase-by-phase plan. Day 0 is already done by me: WSL2 Ubuntu 24.04, NVIDIA driver, apt packages, uv, gh (logged in), and Claude Code are installed.

Work through Phases 1 to 8 in order, autonomously, without asking me for approval. Make reasonable decisions yourself and log them in DECISIONS.md. After each phase, append to PROGRESS.md and commit. Run every training job in the background and poll it with sleep, never in a blocking foreground command. Smoke-test before every long run. Escalate to the fable-debugger subagent only when a trigger in CLAUDE.md fires, and stop only for the big-issue conditions listed there.

Start now with Phase 1.1: create PROGRESS.md and DECISIONS.md with a first dated entry, create the folder layout, initialize git, and verify the GPU is visible to JAX. Report back to me only when the project is finished and pushed to GitHub, or if you hit a big issue.
