# Windows Host Setup Progress — peg-insertion

Target: a WSL2 Ubuntu-24.04 environment where a later Claude Code session can build
`C:\Users\Griff\Projects\peg-insertion`.

Tick a box ONLY after a verification command proved it. Safe to re-run the setup prompt.

## Checklist

- [x] 1. NVIDIA driver >= 580 (596.58)
- [x] 2. WSL2 + Ubuntu-24.04 installed (VERSION 2)
- [x] 3. Linux user `griff` created and default
- [x] 4. C:\Users\Griff\.wslconfig written (memory/processors/swap)
- [x] 5. GPU visible inside WSL (`nvidia-smi` shows RTX 5070 Laptop)
- [x] 6. System apt packages installed
- [x] 7. GitHub CLI (`gh`) installed
- [x] 8. `uv` installed for griff
- [x] 9. Claude Code installed inside WSL for griff
- [x] 10. Git identity configured for griff
- [x] 11. Project `.claude` config files in place (hooks LF + executable)
- [x] 12. Power settings (no sleep on AC) verified; Windows Update pause written to registry (confirm in Settings)

## Notes

### 1. Driver
- `nvidia-smi` on Windows: **Driver Version 596.58**, CUDA 13.2
- GPU: **NVIDIA GeForce RTX 5070 Laptop GPU** (8151 MiB), WDDM
- 596.58 >= 580 -> OK, no driver action needed.

### 4. Memory (.wslconfig)
- `C:\Users\Griff\.wslconfig` written with LF endings:
  `[wsl2]` / `memory=20GB` / `processors=12` / `swap=8GB`
- Replaced a previous config that had memory=26GB, processors=14, swap=8GB.
- Host has 16 logical CPUs, so processors=12 is valid.
- Needs `wsl --shutdown` to take effect (done in sequence below).

### 11. Project config files
- `.claude/` created; full contents of `claude-config/` copied in; `claude-config/` deleted.
- Files present: settings.json, hooks/block-outside-project.sh, hooks/guard-bash.sh,
  agents/fable-debugger.md, agents/opus-debugger.md
- Still pending: CRLF->LF + chmod +x on hooks from WSL, and `bash -n` syntax check.

### 12. Power / Windows Update
- `powercfg /change standby-timeout-ac 0`  -> verified Current AC index 0x00000000
- `powercfg /change monitor-timeout-ac 0`  -> verified Current AC index 0x00000000
- `powercfg /change hibernate-timeout-ac 0`-> verified Current AC index 0x00000000
- Active scheme: Balanced (381b4222-f694-41f0-9685-ff5bb260df2e)
- Windows Update pause written to HKLM\SOFTWARE\Microsoft\WindowsUpdate\UX\Settings:
  - PauseUpdatesExpiryTime       = 2026-10-19T07:18:14Z
  - PauseFeatureUpdatesStartTime = 2026-10-12T07:18:14Z
  - PauseFeatureUpdatesEndTime   = 2026-10-19T07:18:14Z
  - PauseQualityUpdatesStartTime = 2026-10-12T07:18:14Z
  - PauseQualityUpdatesEndTime   = 2026-10-19T07:18:14Z
  - PauseUpdatesPageVisible      = 1 (DWORD)
  Values read back correctly, but the Windows Update service caches its pause state,
  so registry-only pauses are not guaranteed to be honoured => CONFIRM IN SETTINGS (needs me).

### 2. WSL / distro
- `wsl -l -v` initially showed only a pre-existing generic `Ubuntu` (VERSION 2).
- Ran `wsl --install -d Ubuntu-24.04 --no-launch` -> "Distribution successfully installed",
  exit code 0, **no restart required**.
- `wsl -l -v` now lists `Ubuntu-24.04    Stopped    2`.
- The pre-existing `Ubuntu` distro was left untouched (still the default `*` at this point).

### 3. Linux user
- Created non-interactively as root: `useradd -m -s /bin/bash -G sudo griff`
  -> `uid=1000(griff) gid=1000(griff) groups=1000(griff),27(sudo)`
- `/etc/wsl.conf` written with `[user]` / `default=griff` (LF endings, verified with `cat -A`).
- `wsl --terminate Ubuntu-24.04` then `wsl -d Ubuntu-24.04 -- whoami` -> **griff**
- The first-launch / OOBE screen was never needed.
- **No password is set.** If sudo is ever wanted, run from Windows:
  `wsl -d Ubuntu-24.04 -u root -- passwd griff`
  (Until then, use `wsl -d Ubuntu-24.04 -u root -- <cmd>` for root work; it needs no password.)
- Note: invoke WSL from PowerShell, not Git Bash — Git Bash mangles `/etc/...` style
  arguments into Windows paths. Also avoid `printf "...\n"` through PowerShell; the
  backslashes are stripped before bash sees them.

### 5. GPU inside WSL
- `wsl -d Ubuntu-24.04 -- nvidia-smi` works on the first try:
  - NVIDIA-SMI 595.78, **Driver Version 596.58**, CUDA Version 13.2
  - GPU 0: **NVIDIA GeForce RTX 5070 Laptop GPU**, 8151 MiB
- No Linux NVIDIA driver was installed inside WSL, and none should ever be — the
  Windows driver is passed through via /usr/lib/wsl. Installing one would break this.
- Memory config confirmed live after `wsl --shutdown`: Mem total 19Gi, Swap 8.0Gi.

### 11. Project config — verified
All five required files present:
| File | Bytes |
|---|---|
| .claude/settings.json | 1667 (valid JSON) |
| .claude/hooks/block-outside-project.sh | 804 |
| .claude/hooks/guard-bash.sh | 1700 |
| .claude/agents/fable-debugger.md | 2637 |
| .claude/agents/opus-debugger.md | 2032 |

- Ran from WSL: `sed -i 's/\r$//' .../hooks/*.sh && chmod +x .../hooks/*.sh`
- CR bytes remaining (`tr -cd '\015' | wc -c`): **0** in both hooks.
- Permissions inside WSL: `-rwxrwxrwx` for both hooks (the 777 is the DrvFs default for
  /mnt/c without metadata mounting; the executable bit is what matters).
- `bash -n guard-bash.sh` -> exit 0, **no syntax error**
- `bash -n block-outside-project.sh` -> exit 0, no syntax error

### 6. System packages — verified
`apt-get update` + `DEBIAN_FRONTEND=noninteractive apt-get install -y ...` exited **0**,
no `E:`/`W:` lines.

Required verifications:
- `python3.12 --version` -> **Python 3.12.3**
- `jq --version` -> **jq-1.7**
- `ffmpeg -version` -> **ffmpeg version 6.1.1-3ubuntu5**

Also confirmed:
- python3.12-venv working (`python3.12 -m venv --help` OK)
- git 2.43.0, curl 8.5.0, wget 1.21.4, gcc 13.3.0 (build-essential 12.10ubuntu1), patchelf 0.18.0
- libosmesa6 / libosmesa6-dev 25.1.7, libgl1 1.7.0, libglu1-mesa 9.0.2, libglew-dev 2.2.0
  all `install ok installed`

Tooling note: run WSL commands from PowerShell, but write any multi-line bash to an
LF script file first and invoke `wsl -d Ubuntu-24.04 -- bash /mnt/c/.../script.sh`.
PowerShell mangles inline loops, `\n` escapes and `$2`-style awk fields.

### 7. GitHub CLI — verified
Installed via the official cli.github.com apt route:
- keyring -> `/etc/apt/keyrings/githubcli-archive-keyring.gpg` (mode go+r)
- source  -> `/etc/apt/sources.list.d/github-cli.list`
  `deb [arch=amd64 signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main`
- `apt-get update` + `apt-get install -y gh` -> exit 0
- `wsl -d Ubuntu-24.04 -- gh --version` -> **gh version 2.102.0 (2026-09-30)**
- `gh auth login` deliberately NOT run — that is a manual step (see final checklist).

### 8. uv — verified
- Installed as griff: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Landed in `/home/griff/.local/bin` (uv, uvx)
- `~/.bashrc` now has both the installer's `. "$HOME/.local/bin/env"` (line 119) and the
  explicit `export PATH="$HOME/.local/bin:$PATH"` (line 122). Idempotent — re-running
  the setup will report "already present" instead of duplicating.
- `wsl -d Ubuntu-24.04 -u griff -- bash -lc "uv --version"` -> **uv 0.13.0 (x86_64-unknown-linux-gnu)**

### 9. Claude Code inside WSL — verified
- Installed as griff: `curl -fsSL https://claude.ai/install.sh | bash`
- `~/.local/bin/claude` -> `/home/griff/.local/share/claude/versions/2.1.296`
- `wsl -d Ubuntu-24.04 -u griff -- bash -lc "claude --version"` -> **2.1.296 (Claude Code)**
- NOT logged in — login is a manual step (see final checklist).

### 10. Git identity — verified
`wsl -d Ubuntu-24.04 -u griff -- bash -lc "git config --global --list"`:
```
user.name=Griffin
user.email=griffinsamuel869@gmail.com
```

---

## Setup complete — 12 / 12 verified (2026-10-12)

### "Needs me" items (manual)
1. **Confirm Windows Update is paused** — Settings -> Windows Update. Registry values were
   written and read back, but the Update service caches pause state, so verify visually.
2. **`gh auth login`** — intentionally not run.
3. **Claude Code login + trust prompt** — intentionally not run.

### Gotcha for later: the default distro is still the OLD `Ubuntu`
`wsl -l -v` shows `* Ubuntu` as default. A bare `wsl` command opens the WRONG distro.
Always launch **"Ubuntu 24.04"** from the Start menu, or use `wsl -d Ubuntu-24.04`.
To make it the default instead: `wsl --set-default Ubuntu-24.04`
(not done here, to avoid disturbing the pre-existing Ubuntu distro).

### Environment summary
| Thing | Value |
|---|---|
| Windows NVIDIA driver | 596.58 (CUDA 13.2) |
| GPU | RTX 5070 Laptop GPU, 8151 MiB |
| Distro | Ubuntu-24.04, WSL VERSION 2, default user griff (uid 1000, sudo group, no password) |
| WSL resources | 20GB RAM (19Gi seen), 12 processors, 8GB swap |
| Python | 3.12.3 + venv |
| uv | 0.13.0 |
| gh | 2.102.0 (not authenticated) |
| Claude Code | 2.1.296 (not logged in) |
| git identity | Griffin <griffinsamuel869@gmail.com> |
| Project config | .claude/{settings.json, hooks/*.sh (LF, +x), agents/*.md} |
