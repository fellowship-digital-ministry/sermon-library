#!/usr/bin/env python3
"""
Ingest church-website audio sermons that never went up on YouTube.

Background
----------
The normal pipeline (`rss_sermon_downloader.py`) keeps only the church-feed
items that carry a YouTube link, so sermons published as a downloadable MP3
*without* a YouTube video are silently dropped. A one-time crawl found ~335 such
sermons (mostly midweek / Sunday-school teaching). They are catalogued in
`../missing_sermons.csv` (see also `../missing_sermons.md`).

This script ingests a chosen subset of that worklist through the SAME transcribe
-> save -> subtitle -> metadata path the YouTube pipeline uses, the only
difference being step 1: instead of yt-dlp, it fetches the MP3 directly from the
church server. Each ingested sermon gets a synthetic, stable `video_id` with an
`fbc-` prefix (already computed in the worklist), e.g.
`fbc-2022-08-28pm-the-indwelling-of-the-holy-spirit`. That prefix is the single
signal the rest of the system uses to tell "audio-only church sermon" apart from
"YouTube sermon".

Where the output goes (everything keyed by video_id, exactly like YouTube ones):
  - transcription/data/transcripts/{id}.json   (segments)
  - transcription/data/subtitles/{id}.srt|.vtt (the API serves the .srt)
  - transcription/data/metadata/{id}_metadata.json
        carries source="website", audio_url, source_url so the frontend can
        offer "Listen" instead of "Watch on YouTube".
  - appends a row to the embeddings CSV (default video_list.csv) so the next
        `tools/transcript_to_embeddings.py` run pushes it to Pinecone (search/chat).

After this script, run the usual tail of the pipeline (notes -> embeddings ->
catalog), which already discover sermons by scanning the metadata dir / CSV:
    python tools/generate_sermon_notes.py --all --write
    python tools/transcript_to_embeddings.py --skip_existing
    python tools/build_sermon_catalog.py

IMPORTANT — be gentle with the church server
---------------------------------------------
fbcministries.net is a small shared host with little headroom (it returned 503
under even a modest sequential probe). This script downloads ONE file at a time
with a delay between sermons, never in parallel, never re-fetching a file it
already has, and skips (logs) any dead link rather than hammering it. Keep the
delay generous and prefer small batches.

Usage
-----
  # See exactly what WOULD be ingested — no network, no API, no spend:
  python transcription/ingest_website_audio.py --dry-run --status missing --limit 15

  # Real run, gentle, one short series end-to-end (proof batch):
  python transcription/ingest_website_audio.py --series "Ruth Series" --delay 8

  # A handful by id:
  python transcription/ingest_website_audio.py \
      --ids fbc-2022-08-28pm-the-indwelling-of-the-holy-spirit --delay 8
"""
import argparse
import csv
import os
import sys
import time
import logging

import requests

# Paths derived from this file, so we can run --dry-run WITHOUT importing config
# (config.py raises if OPENAI_API_KEY is unset). The heavy imports that need the
# key / network happen lazily, only on a real run.
HERE = os.path.dirname(os.path.abspath(__file__))           # .../transcription
REPO_ROOT = os.path.dirname(HERE)                            # .../sermon-library
DATA_DIR = os.path.join(HERE, "data")
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
TRANSCRIPT_DIR = os.path.join(DATA_DIR, "transcripts")
SUBTITLE_DIR = os.path.join(DATA_DIR, "subtitles")
METADATA_DIR = os.path.join(DATA_DIR, "metadata")
DEFAULT_WORKLIST = os.path.join(REPO_ROOT, "missing_sermons.csv")
DEFAULT_EMBED_CSV = os.path.join(DATA_DIR, "video_list.csv")
UNAVAILABLE_LOG = os.path.join(REPO_ROOT, "missing_sermons_unavailable.csv")

EMBED_CSV_COLUMNS = [
    "video_id", "title", "description", "publish_date", "duration",
    "view_count", "like_count", "url", "thumbnail",
    "processing_status", "processing_date", "transcript_path",
    "embeddings_status", "embeddings_date", "embeddings_count",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("sermon_processing.log"), logging.StreamHandler()],
)
logger = logging.getLogger("ingest_website_audio")


# --------------------------------------------------------------------------- #
# Worklist selection
# --------------------------------------------------------------------------- #
def load_worklist(path):
    if not os.path.exists(path):
        logger.error("Worklist not found: %s", path)
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def select(rows, statuses, series, ids, limit):
    id_set = set(ids or [])
    out = []
    for r in rows:
        if id_set:
            if r["video_id"] in id_set:
                out.append(r)
            continue
        if statuses and r.get("status") not in statuses:
            continue
        if series and (r.get("series") or "").strip().lower() != series.strip().lower():
            continue
        out.append(r)
    # Newest first; the worklist date column is YYYYMMDD.
    out.sort(key=lambda r: r.get("date") or "0", reverse=True)
    if limit:
        out = out[:limit]
    return out


# --------------------------------------------------------------------------- #
# Gentle, one-at-a-time download
# --------------------------------------------------------------------------- #
def download_mp3(url, dest, timeout=120):
    """Fetch a single MP3 to dest. Returns True on success, False on a dead/bad
    link (logged, not retried aggressively). Forces https and verifies the
    response actually looks like audio so a 404 HTML page is never saved as .mp3."""
    https = url.replace("http://", "https://", 1)
    headers = {"User-Agent": "Mozilla/5.0 (sermon-library archival fetch)"}
    for attempt in (1, 2):
        try:
            with requests.get(https, stream=True, timeout=timeout, headers=headers) as resp:
                ctype = resp.headers.get("content-type", "")
                if resp.status_code != 200 or not ctype.lower().startswith("audio"):
                    logger.warning("Unavailable (%s, %s): %s",
                                   resp.status_code, ctype or "no content-type", https)
                    return False
                tmp = dest + ".part"
                with open(tmp, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=65536):
                        if chunk:
                            fh.write(chunk)
                os.replace(tmp, dest)
                size_mb = os.path.getsize(dest) / (1024 * 1024)
                logger.info("Downloaded %.1f MB -> %s", size_mb, dest)
                return True
        except requests.RequestException as e:
            logger.warning("Download attempt %d failed for %s: %s", attempt, https, e)
            if attempt == 1:
                time.sleep(20)  # one gentle pause, then one retry, then give up
    return False


def log_unavailable(row):
    new = not os.path.exists(UNAVAILABLE_LOG)
    with open(UNAVAILABLE_LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["video_id", "date", "title", "mp3_url"])
        w.writerow([row["video_id"], row.get("date", ""), row.get("title", ""), row["mp3_url"]])


# --------------------------------------------------------------------------- #
# Embeddings CSV (the only downstream step that is CSV-driven, not dir-scanned)
# --------------------------------------------------------------------------- #
def append_embed_row(csv_path, vid, title, publish_date, duration, source_url, transcript_path):
    rows = {}
    cols = EMBED_CSV_COLUMNS
    if os.path.exists(csv_path):
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames or EMBED_CSV_COLUMNS
            for r in reader:
                if r.get("video_id"):
                    rows[r["video_id"]] = r
    row = {c: "" for c in cols}
    row.update({
        "video_id": vid, "title": title, "publish_date": publish_date,
        "duration": duration, "url": source_url,
        "processing_status": "processed", "transcript_path": transcript_path,
        "embeddings_status": "pending", "embeddings_count": "0",
    })
    rows[vid] = row  # idempotent upsert by video_id
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows.values():
            w.writerow({c: r.get(c, "") for c in cols})


# --------------------------------------------------------------------------- #
# One sermon, end to end
# --------------------------------------------------------------------------- #
def ingest_one(row, embed_csv, force):
    # Lazy heavy imports (need OPENAI_API_KEY via config) — only on a real run.
    from transcribe_audio import transcribe_audio
    from utils import (save_transcript, save_metadata,
                       generate_srt_file, generate_vtt_file)

    vid = row["video_id"]
    meta_path = os.path.join(METADATA_DIR, f"{vid}_metadata.json")
    if os.path.exists(meta_path) and not force:
        logger.info("Skip (already ingested): %s", vid)
        return "skipped"

    for d in (AUDIO_DIR, TRANSCRIPT_DIR, SUBTITLE_DIR, METADATA_DIR):
        os.makedirs(d, exist_ok=True)

    audio_path = os.path.join(AUDIO_DIR, f"{vid}.mp3")
    if not (os.path.exists(audio_path) and not force):
        if not download_mp3(row["mp3_url"], audio_path):
            log_unavailable(row)
            return "unavailable"

    transcript = transcribe_audio(audio_path)
    if not transcript or not transcript.get("segments"):
        logger.error("Transcription failed/empty: %s", vid)
        return "failed"

    transcript_file = save_transcript(vid, transcript, TRANSCRIPT_DIR)
    generate_srt_file(vid, transcript, SUBTITLE_DIR)
    generate_vtt_file(vid, transcript, SUBTITLE_DIR)

    segs = transcript.get("segments", [])
    duration = int(segs[-1].get("end", 0)) if segs else 0
    source_url = row["mp3_url"].replace("http://", "https://", 1)

    metadata = {
        "video_id": vid,
        "title": row.get("title", vid),
        "publish_date": row.get("date", ""),       # YYYYMMDD, like the YouTube ones
        "source": "website",                         # <-- distinguishes from YouTube
        "service": row.get("service", ""),
        "series": row.get("series", ""),
        "audio_url": source_url,                     # the MP3 (for a "Listen" link)
        "source_url": source_url,                    # church page could replace this later
        "url": source_url,
        "webpage_url": source_url,                   # read by transcript_to_embeddings
        "audio_file": audio_path,
        "transcript_file": transcript_file,
        "duration": duration,
        "transcription_timestamp": __import__("datetime").datetime.now().isoformat(),
        "model": "whisper-1",
    }
    save_metadata(vid, metadata, METADATA_DIR)
    append_embed_row(embed_csv, vid, metadata["title"], metadata["publish_date"],
                     duration, source_url, transcript_file)
    logger.info("Ingested %s (%s, %ds)", vid, metadata["title"], duration)
    return "ingested"


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--worklist", default=DEFAULT_WORKLIST, help="missing_sermons.csv path")
    ap.add_argument("--embed-csv", default=DEFAULT_EMBED_CSV,
                    help="CSV the embeddings step reads (rows appended here)")
    ap.add_argument("--status", default="missing",
                    help="comma-separated worklist statuses to include (default: missing)")
    ap.add_argument("--series", help="only this series (matches the worklist 'series' column)")
    ap.add_argument("--ids", help="comma-separated video_ids (overrides status/series filters)")
    ap.add_argument("--limit", type=int, help="cap how many to ingest this run")
    ap.add_argument("--delay", type=float, default=8.0,
                    help="seconds to wait between sermons (be kind to the server)")
    ap.add_argument("--force", action="store_true", help="re-ingest even if already present")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan only — no download, no API, no spend")
    args = ap.parse_args()

    statuses = [s.strip() for s in args.status.split(",") if s.strip()]
    ids = [s.strip() for s in args.ids.split(",")] if args.ids else None
    rows = select(load_worklist(args.worklist), statuses, args.series, ids, args.limit)

    if not rows:
        logger.info("Nothing matched the selection.")
        return 0

    print(f"\nSelected {len(rows)} sermon(s):\n")
    for r in rows:
        d = r.get("date", "????????")
        dd = f"{d[0:4]}-{d[4:6]}-{d[6:8]}" if len(d) == 8 else d
        print(f"  {dd} [{r.get('service','??'):>2}] {r.get('title','')}")
        print(f"       id:  {r['video_id']}")
        print(f"       mp3: {r['mp3_url']}")
    if args.dry_run:
        print(f"\n[dry-run] Would ingest the above with a {args.delay}s gap between each.")
        print("[dry-run] No files fetched, no transcription, no cost.")
        return 0

    counts = {}
    for i, r in enumerate(rows):
        result = ingest_one(r, args.embed_csv, args.force)
        counts[result] = counts.get(result, 0) + 1
        if args.delay and i < len(rows) - 1:
            time.sleep(args.delay)

    print("\nDone:", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    if counts.get("unavailable"):
        print(f"Dead links logged to: {UNAVAILABLE_LOG}")
    print("\nNext (existing pipeline tail):")
    print("  python tools/generate_sermon_notes.py --all --write")
    print("  python tools/transcript_to_embeddings.py --skip_existing")
    print("  python tools/build_sermon_catalog.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
