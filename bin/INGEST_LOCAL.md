# Local Sermon Ingest

Why this exists: YouTube has progressively blocked unauthenticated yt-dlp from datacenter IPs (GitHub Actions, AWS, etc.), so we moved ingestion off CI and onto a local cron. The Render API still runs in the cloud; only the `discover → transcribe → embed → push` job runs locally.

## What it does

A `systemd --user` timer fires the ingest job **twice daily (09:00 + 21:00 local)**. Each run:

1. `git pull --rebase` the latest state
2. **Discover** any new videos on the YouTube channel via `tools/discover_youtube_backlog.py` (uses `yt-dlp --flat-playlist`; sees the full channel history, not just RSS)
3. **Transcribe** every video marked `pending` in `transcription/data/video_list.csv` via `process_batch.py` (yt-dlp → Whisper API)
4. **Embed** new transcripts into Pinecone via `tools/transcript_to_embeddings.py --skip_existing`
5. **Commit + push** new transcripts/metadata/CSV to the repo. The deployed API picks them up on its next Render rebuild.

The timer uses `Persistent=true`, so if the laptop was off when 09:00 hit, systemd fires the job at next boot. Nothing is lost.

## Prerequisites

| Tool | Why | Install |
|---|---|---|
| **Deno** | yt-dlp needs a JavaScript runtime to solve YouTube's signature challenges | `curl -fsSL https://deno.land/install.sh \| sh -s -- -y` |
| **ffmpeg** | Audio extraction | `sudo apt install ffmpeg` |
| **Python 3.11+** | Pipeline runtime | `sudo apt install python3 python3-venv` |
| **`.env` with API keys** | `OPENAI_API_KEY`, `PINECONE_API_KEY` | copy `.env.example`, fill in |
| **SSH key registered with GitHub** | for the `git push` step from cron | normal `~/.ssh/id_*` config |

## Install

```bash
cd ~/Development/experiments/fellowship/sermon-library
./bin/install_local_ingest.sh
```

The installer:
- verifies the prereqs above
- creates `.venv-ingest/` and installs deps (~250MB)
- copies the systemd units to `~/.config/systemd/user/`
- enables the timer and starts it
- runs `loginctl enable-linger` so the timer fires even when you're not logged in

## Inspecting

```bash
# When will it fire next?
systemctl --user list-timers fellowship-ingest.timer

# Run it right now (skips the schedule)
systemctl --user start fellowship-ingest.service

# Watch the log live
tail -f ~/.local/state/fellowship-ingest.log

# See the last 100 lines
tail -100 ~/.local/state/fellowship-ingest.log

# Service status (last run, exit code, etc.)
systemctl --user status fellowship-ingest.service
```

## Tweaking the schedule

Edit `bin/systemd/fellowship-ingest.timer`, then:

```bash
cp bin/systemd/fellowship-ingest.timer ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user restart fellowship-ingest.timer
```

Common patterns:
- Every 6 hours: `OnCalendar=*-*-* 00/6:00:00`
- Sunday mornings only: `OnCalendar=Sun *-*-* 11:00:00`
- Every hour during the day: `OnCalendar=*-*-* 09..21:00:00`

## Disabling / uninstalling

```bash
# Stop the timer (revert by `systemctl --user start fellowship-ingest.timer`)
systemctl --user stop fellowship-ingest.timer
systemctl --user disable fellowship-ingest.timer

# Full uninstall
systemctl --user disable --now fellowship-ingest.timer
rm ~/.config/systemd/user/fellowship-ingest.{service,timer}
systemctl --user daemon-reload
rm -rf .venv-ingest                                # ~250MB
rm ~/.local/state/fellowship-ingest.log            # history
sudo loginctl disable-linger "$USER"               # optional
```

## Troubleshooting

**`This video is not available` on every video.** Deno isn't on PATH for the cron, or the yt-dlp `remote_components` option got dropped from `transcription/process_batch.py`. Verify with:
```bash
PATH="$HOME/.deno/bin:$PATH" .venv-ingest/bin/yt-dlp --skip-download \
  --remote-components ejs:github \
  --print "%(title)s" "https://www.youtube.com/watch?v=nVKSqmkaEKc"
```
Should print a sermon title. If not, upgrade yt-dlp: `.venv-ingest/bin/pip install --upgrade yt-dlp`.

**`git push` fails from the timer.** Your SSH agent isn't running for the user-level systemd context. Workaround: put your push key in `~/.ssh/id_ed25519` with no passphrase (or use a credential helper). Or: schedule the timer for times you're typically logged in and have an agent.

**Timer never fires when the laptop is off.** That's expected. The `Persistent=true` directive means it fires at next boot instead. To check: `systemctl --user list-timers fellowship-ingest.timer` — the `LAST` column is the most recent actual run.

**Whisper API costs more than expected.** Each ingest run logs the CSV state ("Processing X of Y pending"). At ~$0.006/min × ~45-min sermons, expect ~$0.27/sermon. If you see runaway costs, check that `processing_status` is being updated correctly in `transcription/data/video_list.csv` — stuck `pending` rows get reprocessed every run.

## What's NOT here

- The keep-warm GH Actions workflow (`.github/workflows/keep_api_warm.yml`) is still alive and unrelated — it just pings Render's `/health` every 10 min from GH's IPs (an HTTP ping isn't subject to YouTube's blocks).
- The Render API service is unchanged. It reads transcripts/metadata from the committed files and embeddings from Pinecone.
