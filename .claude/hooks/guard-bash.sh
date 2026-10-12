#!/usr/bin/env bash
# PreToolUse hook for Bash: block commands that would leave the project folder,
# change the system, or touch credentials. Exit 2 blocks; stderr goes to Claude.
set -u
INPUT="$(cat)"
CMD="$(printf '%s' "$INPUT" | jq -r '.tool_input.command // empty')"
[ -z "$CMD" ] && exit 0

block() { echo "BLOCKED by guard-bash hook: $1 (see CLAUDE.md, Hard boundaries). Command was: $CMD" >&2; exit 2; }

# System changes / privilege escalation (anywhere in a pipeline or chain).
if printf '%s' "$CMD" | grep -Eq '(^|[;&|(] *)(sudo|su|doas|apt|apt-get|dpkg|snap|systemctl|shutdown|reboot|mkfs|mount|umount)( |$)'; then
  block "system-level command"
fi
# Global git config or credential files.
if printf '%s' "$CMD" | grep -Eq -- '--global( |$)|--system( |$)|/\.ssh(/|$| )|/\.config/gh|(~|\$HOME|/home/[^/ ]+|/root)/\.claude(/|$| )|\.gnupg|\.netrc|\.git-credentials'; then
  block "touches global config or credentials"
fi
# Explicit paths into other user folders on the Windows drive.
if printf '%s' "$CMD" | grep -Eq '/mnt/c/Users/Griff/(AppData|Documents|Downloads|Desktop|Pictures|Videos|Music|OneDrive)'; then
  block "path outside the project folder"
fi
# rm against anything that is not clearly inside the project.
if printf '%s' "$CMD" | grep -Eq '(^|[;&|] *)rm +(-[a-zA-Z]+ +)*(/|~|\$HOME|\.\./)'; then
  case "$CMD" in
    *"/mnt/c/Users/Griff/Projects/peg-insertion/"*) : ;;
    *) block "rm on a path outside the project" ;;
  esac
fi
# Force pushes / history rewrites, even when wrapped.
if printf '%s' "$CMD" | grep -Eq 'git +push[^;&|]*( -f| --force)|git +reset +--hard|git +clean|git +branch +-D|gh +repo +(delete|edit)'; then
  block "destructive git/gh operation"
fi
exit 0
