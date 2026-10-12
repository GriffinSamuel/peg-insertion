#!/usr/bin/env bash
# PreToolUse hook: block Edit/Write/MultiEdit/NotebookEdit outside the project root.
# Exit 2 blocks the tool call; stderr is shown to Claude.
set -u
INPUT="$(cat)"
TARGET="$(printf '%s' "$INPUT" | jq -r '.tool_input.file_path // .tool_input.notebook_path // empty')"
[ -z "$TARGET" ] && exit 0

ROOT="$(realpath -m "${CLAUDE_PROJECT_DIR:-$PWD}")"
CWD="$(printf '%s' "$INPUT" | jq -r '.cwd // empty')"
if [ -n "$CWD" ] && [ "${TARGET#/}" = "$TARGET" ]; then
  RESOLVED="$(realpath -m "$CWD/$TARGET")"
else
  RESOLVED="$(realpath -m "$TARGET")"
fi

case "$RESOLVED" in
  "$ROOT"|"$ROOT"/*) exit 0 ;;
  *)
    echo "BLOCKED by project boundary hook: '$RESOLVED' is outside '$ROOT'. All files must stay inside the project folder (see CLAUDE.md, Hard boundaries)." >&2
    exit 2 ;;
esac
