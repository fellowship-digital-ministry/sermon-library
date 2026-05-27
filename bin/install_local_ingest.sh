#!/usr/bin/env bash
# One-shot installer for the local sermon-ingest cron.
# Idempotent — safe to re-run.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "==> Installing fellowship-ingest local cron"
echo "    repo: $REPO_ROOT"
echo ""

# --- pre-flight checks ---
echo "==> Checking prerequisites..."

command -v deno >/dev/null 2>&1 || {
  echo "    !!  Deno not found. Install with:"
  echo "        curl -fsSL https://deno.land/install.sh | sh -s -- -y"
  echo "        (yt-dlp needs Deno to solve YouTube's JS challenges.)"
  exit 1
}
echo "    ok  deno: $(deno --version | head -1)"

command -v ffmpeg >/dev/null 2>&1 || {
  echo "    !!  ffmpeg not found. Install with: sudo apt install ffmpeg"
  exit 1
}
echo "    ok  ffmpeg: $(ffmpeg -version 2>&1 | head -1 | cut -d' ' -f1-3)"

[[ -f "$REPO_ROOT/.env" ]] || {
  echo "    !!  $REPO_ROOT/.env not found. Copy .env.example and fill in the keys."
  exit 1
}
echo "    ok  .env present"

[[ -d "$REPO_ROOT/.venv-ingest" ]] || {
  echo "    --  Creating .venv-ingest..."
  python3 -m venv "$REPO_ROOT/.venv-ingest"
  "$REPO_ROOT/.venv-ingest/bin/pip" install --upgrade pip
  "$REPO_ROOT/.venv-ingest/bin/pip" install -r "$REPO_ROOT/requirements.txt"
  "$REPO_ROOT/.venv-ingest/bin/pip" install pandas yt-dlp python-dotenv
}
echo "    ok  .venv-ingest present"

# --- make wrapper executable ---
chmod +x "$REPO_ROOT/bin/run_ingest.sh"

# --- install systemd unit files ---
UNIT_DIR="$HOME/.config/systemd/user"
mkdir -p "$UNIT_DIR"

echo ""
echo "==> Installing systemd unit files to $UNIT_DIR"
cp "$SCRIPT_DIR/systemd/fellowship-ingest.service" "$UNIT_DIR/"
cp "$SCRIPT_DIR/systemd/fellowship-ingest.timer"   "$UNIT_DIR/"

systemctl --user daemon-reload
systemctl --user enable --now fellowship-ingest.timer

# --- allow service to run when not logged in ---
echo ""
echo "==> Enabling user lingering (so the timer fires even when you're not logged in)"
if loginctl show-user "$USER" 2>/dev/null | grep -q "Linger=yes"; then
  echo "    ok  already enabled"
else
  echo "    Running: sudo loginctl enable-linger $USER"
  sudo loginctl enable-linger "$USER"
fi

echo ""
echo "==> Done."
echo ""
echo "Status:"
systemctl --user status fellowship-ingest.timer --no-pager 2>/dev/null | head -12

echo ""
echo "Next steps:"
echo "  - Inspect upcoming firings:  systemctl --user list-timers fellowship-ingest.timer"
echo "  - Run manually right now:    systemctl --user start fellowship-ingest.service"
echo "  - Watch the log live:        tail -f ~/.local/state/fellowship-ingest.log"
echo "  - Disable temporarily:       systemctl --user stop fellowship-ingest.timer"
echo "  - Uninstall:                 see bin/INGEST_LOCAL.md"
