#!/usr/bin/env python3
"""
Fill in missing sermon publish dates from the YouTube channel RSS feed.

The local ingest does not capture a video's upload date (yt-dlp is bot-gated
from many IPs, and the podcast feed's pubDate is the post date, not the sermon
date), so recently ingested sermons land with an empty publish_date and sort to
the bottom of the catalog as "Date unknown". The channel's Atom feed
(https://www.youtube.com/feeds/videos.xml?channel_id=...) carries the real
<published> date for the most recent ~15 uploads and is plain HTTP — not gated.

This tool maps video_id -> upload date (YYYYMMDD) from that feed and writes it
into any metadata file whose publish_date is missing/NaN. It only covers what
the feed currently holds (the newest sermons), which is exactly the set that
matters for "newest first". Run it in the ingest pipeline after transcription so
each week's new sermons get dated.

    python tools/backfill_dates_from_rss.py            # default channel
    python tools/backfill_dates_from_rss.py --channel-id UCxxxx
"""

import argparse
import glob
import json
import os
import re
import xml.etree.ElementTree as ET

import requests

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METADATA_DIR = os.path.join(REPO_ROOT, "transcription", "data", "metadata")
DEFAULT_CHANNEL_ID = "UCek_LI7dZopFJEvwxDnovJg"  # Fellowship Baptist Church Oakton, VA
NS = {"a": "http://www.w3.org/2005/Atom", "yt": "http://www.youtube.com/xml/schemas/2015"}


def fetch_feed_dates(channel_id: str) -> dict:
    """Return {video_id: 'YYYYMMDD'} from the channel Atom feed."""
    url = "https://www.youtube.com/feeds/videos.xml?channel_id=" + channel_id
    xml = requests.get(url, timeout=30).text
    root = ET.fromstring(xml)
    out = {}
    for entry in root.findall("a:entry", NS):
        vid = entry.findtext("yt:videoId", default="", namespaces=NS)
        published = entry.findtext("a:published", default="", namespaces=NS)  # ISO 8601
        if vid and len(published) >= 10:
            out[vid] = published[:10].replace("-", "")  # YYYYMMDD
    return out


def has_valid_date(meta: dict) -> bool:
    return bool(re.fullmatch(r"\d{8}", str(meta.get("publish_date") or "")))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--channel-id", default=DEFAULT_CHANNEL_ID)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    dates = fetch_feed_dates(args.channel_id)
    print(f"Channel feed provided dates for {len(dates)} recent videos.")

    filled = 0
    for path in glob.glob(os.path.join(METADATA_DIR, "*_metadata.json")):
        try:
            meta = json.load(open(path, encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        vid = meta.get("video_id")
        if has_valid_date(meta) or vid not in dates:
            continue
        print(f"  {vid}: publish_date -> {dates[vid]}  ({meta.get('title','')[:45]})")
        filled += 1
        if not args.dry_run:
            meta["publish_date"] = dates[vid]
            with open(path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n{'Would fill' if args.dry_run else 'Filled'} {filled} sermon date(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
