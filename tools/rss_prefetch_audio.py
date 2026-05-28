#!/usr/bin/env python3
"""Pre-download sermon audio from the church's podcast RSS feed.

YouTube downloads have become unreliable (JS-challenge gating, IP blocks),
but the church's podcast feed at fbcministries.net/feed/podcast hosts the
same sermons as plain-HTTP MP3s. Each podcast item embeds the matching
YouTube video ID in its <itunes:subtitle> / <itunes:summary> field, so we
can join by ID and drop the mp3 into transcription/data/audio/{id}.mp3.

process_batch.py's download_audio() short-circuits when the audio file
already exists (process_batch.py:59), so this script "feeds" it: anything
we pre-fetch here skips the yt-dlp path entirely. Sermons NOT in the RSS
window (only the ~10 most recent) fall through to yt-dlp as before.

Idempotent: a sermon is considered "have it" if either its transcript or
its audio file already exists locally. The feed is small (~20 KB) and
each mp3 is ~10-30 MB, so this is cheap to run on every cron tick.

Usage:
  python tools/rss_prefetch_audio.py            # download missing
  python tools/rss_prefetch_audio.py --dry-run  # just report
  python tools/rss_prefetch_audio.py --feed-url https://example.com/feed
"""
import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Iterator, Optional, Tuple
from xml.etree import ElementTree as ET

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
AUDIO_DIR = REPO_ROOT / "transcription" / "data" / "audio"
TRANSCRIPT_DIR = REPO_ROOT / "transcription" / "data" / "transcripts"
DEFAULT_FEED = "https://fbcministries.net/feed/podcast"

ITUNES_NS = {"itunes": "http://www.itunes.com/dtds/podcast-1.0.dtd"}

# YouTube IDs are exactly 11 chars of [A-Za-z0-9_-]. Anchoring on that
# avoids picking up tracking params or non-YouTube strings that happen to
# follow a youtube.com prefix.
YOUTUBE_ID_RE = re.compile(
    r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/|"
    r"youtube\.com/v/|youtube\.com/shorts/)([A-Za-z0-9_-]{11})"
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("rss-prefetch")


def extract_youtube_id(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    m = YOUTUBE_ID_RE.search(text)
    return m.group(1) if m else None


def parse_items(xml: str) -> Iterator[Tuple[str, str, str]]:
    """Yield (video_id, mp3_url, title) for each podcast item that has both."""
    root = ET.fromstring(xml)
    for item in root.findall(".//item"):
        encl = item.find("enclosure")
        mp3 = encl.get("url") if encl is not None else None
        if not mp3:
            continue
        video_id = None
        for finder in (
            lambda: item.find("itunes:subtitle", ITUNES_NS),
            lambda: item.find("itunes:summary", ITUNES_NS),
            lambda: item.find("description"),
        ):
            el = finder()
            if el is not None and el.text:
                video_id = extract_youtube_id(el.text)
                if video_id:
                    break
        if not video_id:
            continue
        title_el = item.find("title")
        title = (title_el.text or "") if title_el is not None else ""
        yield video_id, mp3, title


def already_have(video_id: str) -> bool:
    """We 'have' a sermon if either its transcript or audio file exists."""
    return (
        (TRANSCRIPT_DIR / f"{video_id}.json").exists()
        or (AUDIO_DIR / f"{video_id}.mp3").exists()
    )


def download(mp3_url: str, dest: Path) -> int:
    """Stream mp3 to dest via .tmp + atomic rename. Returns bytes written."""
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    total = 0
    with requests.get(mp3_url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(8192):
                if chunk:
                    f.write(chunk)
                    total += len(chunk)
    tmp.replace(dest)
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--feed-url", default=DEFAULT_FEED,
                        help=f"Podcast RSS feed URL (default: {DEFAULT_FEED})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would be downloaded without doing it")
    args = parser.parse_args()

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    log.info(f"Fetching feed: {args.feed_url}")
    try:
        xml = requests.get(args.feed_url, timeout=30).text
    except requests.RequestException as e:
        log.error(f"Feed fetch failed: {e}")
        return 1

    items = list(parse_items(xml))
    log.info(f"RSS items with mp3 + matchable YouTube ID: {len(items)}")
    if not items:
        return 0

    needed = [t for t in items if not already_have(t[0])]
    log.info(f"  already have (transcript or audio): {len(items) - len(needed)}")
    log.info(f"  need to download: {len(needed)}")

    if args.dry_run:
        for vid, _mp3, title in needed:
            log.info(f"  WOULD download {vid}  {title[:70]}")
        return 0

    ok = fail = 0
    for vid, mp3, title in needed:
        dest = AUDIO_DIR / f"{vid}.mp3"
        log.info(f"Downloading {vid}: {title[:70]}")
        try:
            n = download(mp3, dest)
            log.info(f"  wrote {n/1024/1024:.1f} MB to {dest}")
            ok += 1
        except Exception as e:
            log.error(f"  failed: {e}")
            fail += 1
    log.info(f"Done. downloaded={ok}, failed={fail}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
