#!/usr/bin/env bash
# Run the full sermon ingest pipeline: discover → transcribe → embed → push.
#
# Intended to be invoked by the systemd timer (bin/systemd/fellowship-ingest.timer)
# but works fine when run by hand for testing.
#
# Self-contained: sets its own PATH, sources its own .env, activates its own venv.
# Designed to survive cron's minimal environment.

set -uo pipefail  # NOT -e — we want to keep going if discovery finds nothing

# --- locate repo (this script lives at REPO/bin/run_ingest.sh) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# --- logging ---
LOG_DIR="$HOME/.local/state"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/fellowship-ingest.log"
exec >> "$LOG_FILE" 2>&1
echo ""
echo "================================================================"
echo "fellowship-ingest run starting at $(date -Iseconds)"
echo "repo: $REPO_ROOT"
echo "================================================================"

# --- environment ---
# systemd timers inherit a minimal PATH; we must include everywhere our
# dependencies live: Deno (for yt-dlp JS challenge), system bins (ffmpeg, git).
export PATH="$HOME/.deno/bin:/usr/local/bin:/usr/bin:/bin"

# Load API keys from .env (gitignored). Required: OPENAI_API_KEY, PINECONE_API_KEY.
if [[ -f .env ]]; then
  set -a; . ./.env; set +a
else
  echo "ERROR: .env not found at $REPO_ROOT/.env — cannot continue"
  exit 1
fi

# Disable POC_MODE for ingest runs (it was a per-run cap for the old GH backfill).
export POC_MODE=false

# Activate the dedicated ingest venv.
if [[ ! -d .venv-ingest ]]; then
  echo "ERROR: .venv-ingest not found. Run: python3 -m venv .venv-ingest && .venv-ingest/bin/pip install -r requirements.txt && .venv-ingest/bin/pip install pandas yt-dlp python-dotenv"
  exit 1
fi
source .venv-ingest/bin/activate

# Sanity check: Deno must be on PATH for yt-dlp's JS challenge solver.
if ! command -v deno >/dev/null 2>&1; then
  echo "WARNING: deno not found on PATH. yt-dlp will likely fail with 'video not available'. Install: curl -fsSL https://deno.land/install.sh | sh -s -- -y"
fi

# --- pull latest first so our commits don't race with anything else ---
echo ""
echo "[1/5] git pull"
git pull --rebase --autostash origin main || { echo "git pull failed; aborting"; exit 2; }

# --- 1. discover new videos via yt-dlp flat-playlist enumeration ---
echo ""
echo "[2/5] discover new videos from YouTube channel"
python tools/discover_youtube_backlog.py || echo "(discover step had issues, continuing)"

# --- 2. transcribe everything pending ---
echo ""
echo "[3/5] transcribe pending sermons"
( cd transcription && python process_batch.py --csv data/video_list.csv --full --delay 3 )
TRANSCRIBE_STATUS=$?
echo "transcribe exit status: $TRANSCRIBE_STATUS"

# --- 3. push new transcripts to Pinecone ---
echo ""
echo "[4/5] generate embeddings for new transcripts"
python tools/transcript_to_embeddings.py \
  --video_list_csv transcription/data/video_list.csv \
  --transcript_dir transcription/data/transcripts \
  --skip_existing || echo "(embeddings step had issues, continuing)"

# --- 4. commit and push results ---
echo ""
echo "[5/5] commit & push"
git add transcription/data/video_list.csv \
        transcription/data/transcripts/ \
        transcription/data/metadata/ \
        transcription/data/subtitles/ 2>/dev/null || true

if git diff --staged --quiet; then
  echo "Nothing to commit."
else
  TODAY=$(date -u +%Y-%m-%d)
  git commit -m "Local ingest: $TODAY [skip ci]" \
             --author="Fellowship Ingest Bot <fellowship-ingest@local>"
  git push origin main || echo "git push failed — will retry next run"
fi

echo ""
echo "Run complete at $(date -Iseconds)"
