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
echo "[1/8] git pull"
git pull --rebase --autostash origin main || { echo "git pull failed; aborting"; exit 2; }

# --- discover new videos via yt-dlp flat-playlist enumeration ---
# Metadata-only; this path is NOT subject to YouTube's download gating.
echo ""
echo "[2/8] discover new videos from YouTube channel"
python tools/discover_youtube_backlog.py || echo "(discover step had issues, continuing)"

# --- prefetch audio from the church podcast RSS feed ---
# YouTube downloads are increasingly blocked, but the church mirrors each
# sermon as plain-HTTP MP3 on its podcast feed. Drop those into the audio
# dir; process_batch.py's "file already exists" guard then skips yt-dlp
# entirely. Only the most recent ~10 sermons are in the RSS window, so
# anything older still falls through to yt-dlp.
echo ""
echo "[3/8] prefetch audio from church podcast RSS"
python tools/rss_prefetch_audio.py || echo "(RSS prefetch had issues, continuing)"

# --- transcribe everything pending ---
# --delay 15: yesterday's --delay 3 burst hammered yt-dlp's player-JSON
# extraction (and probably YouTube's per-IP burst protection), causing 31
# sermons to fail with "Requested format is not available". Individual
# retries succeed cleanly, so the failure was throughput-driven.
echo ""
echo "[4/8] transcribe pending sermons"
( cd transcription && python process_batch.py --csv data/video_list.csv --full --delay 15 )
TRANSCRIBE_STATUS=$?
echo "transcribe exit status: $TRANSCRIBE_STATUS"

# --- push new transcripts to Pinecone ---
echo ""
echo "[5/8] generate embeddings for new transcripts"
python tools/transcript_to_embeddings.py \
  --video_list_csv transcription/data/video_list.csv \
  --transcript_dir transcription/data/transcripts \
  --skip_existing || echo "(embeddings step had issues, continuing)"

# --- extract Bible references from new transcripts ---
# Tracker file (transcription/data/bible_references/processed_files.json)
# already handles incremental — skips transcripts it has already seen.
# Uses Claude Haiku 4.5 via OpenRouter (OPENROUTER_API_KEY in .env).
echo ""
echo "[6/8] extract bible references for new transcripts"
python tools/bible_reference_extractor.py \
  --input-dir transcription/data/transcripts \
  --output-dir transcription/data/bible_references \
  || echo "(bible reference extraction had issues, continuing)"

# --- generate "In this sermon" footnote summaries for new references ---
# Idempotent: only refs without a point_summary field get processed, so this
# is cheap on weekly runs (only the new sermon's refs). Uses Sonnet 4.6 by
# default for theological nuance; override with SUMMARY_MODEL env var.
echo ""
echo "[7/8] generate footnote summaries for new references"
python tools/generate_reference_summaries.py \
  || echo "(footnote summary generation had issues, continuing)"

# --- commit and push results ---
echo ""
echo "[8/8] commit & push"
git add transcription/data/video_list.csv \
        transcription/data/transcripts/ \
        transcription/data/metadata/ \
        transcription/data/subtitles/ \
        transcription/data/bible_references/ 2>/dev/null || true

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
