How to use this file: open PowerShell **as Administrator** (right-click → Run as administrator), then run:

    cd C:\Users\Griff\Projects
    claude

(If `claude` is not found on Windows, open VS Code as Administrator, open the folder C:\Users\Griff\Projects, and use the Claude Code panel instead.) Paste everything below the line as your first message. If Windows needs a reboot partway through, reboot and paste the same message again; it skips finished steps.

---

You are setting up my Windows laptop so that a separate Claude Code session inside WSL2 can later build the project in C:\Users\Griff\Projects\peg-insertion. Do every step below that can be done from here, in order. Stop only at steps marked "needs me", and when you stop, tell me exactly what to do and what to paste next. Ignore any CLAUDE.md you encounter under Projects\peg-insertion; those rules are for the later WSL build session, not for this setup. Do not touch anything under Projects\gcode-rag. Keep a checklist in C:\Users\Griff\Projects\peg-insertion\SETUP_PROGRESS.md: tick each step only after a verification command proves it, and skip any step already ticked, because I may re-paste this prompt after a reboot. Long installs should run in the background and be polled, not run in a single blocking call.

1. Driver. Run `nvidia-smi` and record the driver version in SETUP_PROGRESS.md. If it is below 580, tell me to install the latest GeForce driver from the NVIDIA App and reboot (needs me), and carry on with the remaining steps anyway.

2. WSL. Run `wsl -l -v`. If Ubuntu-24.04 is not listed with VERSION 2, run `wsl --install -d Ubuntu-24.04 --no-launch`. If Windows says a restart is required, tick the finished steps, tell me to reboot and re-paste this prompt, and stop.

3. Linux user. Try to create the user without the interactive first-launch screen: run as root inside the distro `id griff || useradd -m -s /bin/bash -G sudo griff`, then write /etc/wsl.conf containing `[user]` and `default=griff`, then `wsl --terminate Ubuntu-24.04`. Verify with `wsl -d Ubuntu-24.04 -- whoami` printing `griff`. Do not set a password; note in SETUP_PROGRESS.md that `wsl -d Ubuntu-24.04 -u root -- passwd griff` sets one later if I ever want sudo. If WSL insists on its first-launch setup screen, tell me to open "Ubuntu 24.04" from the Start menu once, create a user named griff, close it, and re-paste this prompt (needs me).

4. Memory. Write C:\Users\Griff\.wslconfig with these three lines under a `[wsl2]` header: `memory=20GB`, `processors=12`, `swap=8GB`. Then run `wsl --shutdown`.

5. GPU inside WSL. `wsl -d Ubuntu-24.04 -- nvidia-smi` must show the RTX 5070 Laptop GPU. Never install a Linux NVIDIA driver inside WSL; the Windows driver is the only one. If it fails, run `wsl --update` then `wsl --shutdown` and retry once; if it still fails, record the error and tell me (needs me; usually a driver update or reboot).

6. System packages, as root inside the distro, non-interactive: `apt-get update` then `DEBIAN_FRONTEND=noninteractive apt-get install -y git curl wget jq ffmpeg build-essential python3.12 python3.12-venv libosmesa6 libosmesa6-dev libgl1 libglu1-mesa libglew-dev patchelf`. Verify `python3.12 --version`, `jq --version`, `ffmpeg -version`.

7. GitHub CLI, as root inside the distro, following the official apt instructions from cli.github.com (keyring to /etc/apt/keyrings, sources list entry, `apt-get update`, `apt-get install -y gh`). Verify `wsl -d Ubuntu-24.04 -- gh --version`. Do NOT run `gh auth login`; I do that myself later.

8. uv, as user griff with a login shell: `curl -LsSf https://astral.sh/uv/install.sh | sh`, then make sure `export PATH="$HOME/.local/bin:$PATH"` is in ~/.bashrc. Verify `wsl -d Ubuntu-24.04 -u griff -- bash -lc "uv --version"`.

9. Claude Code inside WSL, as griff: `curl -fsSL https://claude.ai/install.sh | bash`. Verify `wsl -d Ubuntu-24.04 -u griff -- bash -lc "claude --version"`. Do not try to log it in; I do that myself.

10. Git identity inside WSL, as griff, global: user.name "Griffin", user.email "griffinsamuel869@gmail.com". Verify with `git config --global --list`.

11. Project config files. In C:\Users\Griff\Projects\peg-insertion there is a folder named claude-config. Create the folder .claude there, copy the full contents of claude-config into .claude (preserving the agents and hooks subfolders), then delete claude-config. Then from WSL run `sed -i 's/\r$//' /mnt/c/Users/Griff/Projects/peg-insertion/.claude/hooks/*.sh && chmod +x /mnt/c/Users/Griff/Projects/peg-insertion/.claude/hooks/*.sh`. Verify these five files exist: .claude/settings.json, .claude/hooks/block-outside-project.sh, .claude/hooks/guard-bash.sh, .claude/agents/fable-debugger.md, .claude/agents/opus-debugger.md, and that `wsl -d Ubuntu-24.04 -- bash -n /mnt/c/Users/Griff/Projects/peg-insertion/.claude/hooks/guard-bash.sh` reports no syntax error.

12. Power. Run `powercfg /change standby-timeout-ac 0`, `powercfg /change monitor-timeout-ac 0`, `powercfg /change hibernate-timeout-ac 0`. Then try to pause Windows Update for 7 days through the registry (the PauseUpdatesExpiryTime, PauseFeatureUpdatesStartTime/EndTime and PauseQualityUpdatesStartTime/EndTime values under HKLM\SOFTWARE\Microsoft\WindowsUpdate\UX\Settings). If that fails or you are unsure it took effect, tell me to pause updates in Settings → Windows Update (needs me).

13. Finish. Print a short summary: the driver version, whether every step is ticked, and the list of "needs me" items in order. Then print this exact checklist for me to do in an Ubuntu 24.04 terminal, each command in its own code block:
   a. `gh auth login` (choose GitHub.com, HTTPS, answer Y to "Authenticate Git with your GitHub credentials?", then browser login)
   b. `gh auth setup-git && gh auth status`
   c. `cd /mnt/c/Users/Griff/Projects/peg-insertion && claude` (accept the trust prompt and log in)
   d. In Claude Code: `/model`, pick Opus, and note whether Fable is listed
   e. Send once: `Use the fable-debugger subagent to reply with the single word OK, then stop.` and answer any billing prompt
   f. Open START_PROMPT.md, copy everything below its line, paste it as the message (if Fable was not listed, add the sentence noted at the top of that file)
   g. Keep the laptop plugged in; check the terminal morning and evening; if it is idle with no DONE.md, paste the contents of RESUME_PROMPT.md
