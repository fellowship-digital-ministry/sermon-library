#!/usr/bin/env python3
"""Enumerate a YouTube channel's full upload history and append any missing
videos to video_list.csv with processing_status=pending.

Why this exists: rss_sermon_downloader.py discovers new videos via YouTube's
RSS feed, which only returns the ~15 most recent uploads. After any extended
downtime, hundreds of videos can fall off the RSS window and become invisible
to the rest of the pipeline. This script uses yt-dlp's --flat-playlist mode
(metadata only, no audio download) to walk the entire channel uploads playlist,
then appends any video_ids not already in the CSV as pending rows. The normal
transcribe→embed pipeline picks them up on the next run.

Idempotent. No LLM cost. Typically ~30s for a 500-video channel.

Usage:
  python tools/discover_youtube_backlog.py --dry-run        # report only
  python tools/discover_youtube_backlog.py                   # update CSV
  python tools/discover_youtube_backlog.py --channel-id UCxx # different channel
"""
import argparse
import csv
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set

try:
    import yt_dlp
except ImportError:
    sys.exit("yt-dlp not installed. Run: pip install yt-dlp")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("backfill")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = REPO_ROOT / "transcription" / "data" / "video_list.csv"
DEFAULT_CHANNEL_ID = "UCek_LI7dZopFJEvwxDnovJg"  # Fellowship Baptist Church Oakton, VA

# Must match the column order rss_sermon_downloader.py writes — anything
# missing here will end up blank for new rows, anything extra will be ignored.
CSV_COLUMNS = [
    "video_id", "title", "description", "publish_date",
    "duration", "view_count", "like_count", "url", "thumbnail",
    "processing_status", "processing_date", "transcript_path",
    "embeddings_status", "embeddings_date", "embeddings_count",
]


def load_existing_ids(csv_path: Path) -> Set[str]:
    if not csv_path.exists():
        log.warning(f"CSV does not exist at {csv_path} — will create on append")
        return set()
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {row["video_id"] for row in reader if row.get("video_id")}


def enumerate_channel(channel_id: str) -> List[Dict]:
    """List every video in the channel's uploads tab via flat-playlist."""
    channel_url = f"https://www.youtube.com/channel/{channel_id}/videos"
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        # extract_flat=in_playlist returns one entry per video with id/title
        # but does NOT fetch each video's individual page (much faster).
        "extract_flat": "in_playlist",
        "skip_download": True,
        "ignoreerrors": True,
    }
    log.info(f"Enumerating channel: {channel_url}")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(channel_url, download=False)

    entries = (info or {}).get("entries") or []
    log.info(f"yt-dlp returned {len(entries)} entries")

    rows = []
    for e in entries:
        if not e or not e.get("id"):
            continue
        video_id = e["id"]
        # flat-playlist sometimes gives upload_date as YYYYMMDD, sometimes a timestamp.
        publish_date = e.get("upload_date") or ""
        if not publish_date and e.get("timestamp"):
            publish_date = datetime.utcfromtimestamp(e["timestamp"]).strftime("%Y%m%d")
        rows.append({
            "video_id": video_id,
            "title": e.get("title") or f"Unknown ({video_id})",
            "description": "",
            "publish_date": publish_date,
            "duration": e.get("duration") or 0,
            "view_count": e.get("view_count") or 0,
            "like_count": 0,
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "thumbnail": f"https://i3.ytimg.com/vi/{video_id}/hqdefault.jpg",
            "processing_status": "pending",
            "processing_date": "",
            "transcript_path": "",
            "embeddings_status": "pending",
            "embeddings_date": "",
            "embeddings_count": "0",
        })
    return rows


def append_rows(csv_path: Path, new_rows: List[Dict]) -> int:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    is_new_file = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        if is_new_file:
            writer.writeheader()
        for row in new_rows:
            writer.writerow(row)
    return len(new_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_CSV,
                        help=f"Path to video_list.csv (default: {DEFAULT_CSV})")
    parser.add_argument("--channel-id", default=DEFAULT_CHANNEL_ID,
                        help=f"YouTube channel ID (default: {DEFAULT_CHANNEL_ID})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report counts but don't modify the CSV")
    args = parser.parse_args()

    existing_ids = load_existing_ids(args.csv_path)
    log.info(f"CSV currently tracks {len(existing_ids)} video IDs")

    all_videos = enumerate_channel(args.channel_id)
    log.info(f"Channel currently exposes {len(all_videos)} videos via yt-dlp")

    missing = [v for v in all_videos if v["video_id"] not in existing_ids]
    log.info(f"Missing from CSV: {len(missing)} videos")

    if missing:
        preview = missing[:10]
        log.info(f"First {len(preview)} missing (newest first per yt-dlp ordering):")
        for v in preview:
            log.info(f"  {v['video_id']}  {v['publish_date'] or '????????'}  {v['title'][:70]}")

    if args.dry_run:
        log.info("Dry-run: not modifying CSV")
        return 0

    if missing:
        appended = append_rows(args.csv_path, missing)
        log.info(f"Appended {appended} pending rows to {args.csv_path}")
    else:
        log.info("Nothing to append")
    return 0


if __name__ == "__main__":
    sys.exit(main())
