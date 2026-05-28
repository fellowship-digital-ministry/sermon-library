#!/usr/bin/env python3
"""
Build the static sermon catalog the public Sermons page browses.

Reads every sermon metadata file and emits a single JSON array (newest first)
that GitHub Pages serves directly — so browsing the library is fully static:
instant, free, and works even when the API/Render is asleep. Semantic search
still goes through the API; only the catalog browse is static.

Each entry carries the generated `notes` block (when present) so BOTH the
Sermons list (uses `description`) and the transcript page's notes card read
from this one file — no API call needed to show notes.

    python tools/build_sermon_catalog.py \
        --out ../fellowship-digital-ministry.github.io/assets/data/sermons_catalog.json

Run at the end of the ingest pipeline (after notes generation) and on backfill.

NOTE on scale: with full notes for ~575 sermons this file is well under 1 MB,
fine for a one-time static fetch. If it ever grows uncomfortable, split into a
light list (no notes) for the index + per-sermon notes files; the page code
already keys everything by video_id, so that change is localized.
"""

import argparse
import glob
import json
import os
import re


def normalize_date(value) -> str:
    """Return a clean YYYYMMDD string, or '' for missing/NaN/garbage.
    Some metadata files carry publish_date as NaN or a non-date string; those
    must sort LAST (treated as unknown), not first."""
    s = str(value or "").strip()
    return s if re.fullmatch(r"\d{8}", s) else ""

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METADATA_DIR = os.path.join(REPO_ROOT, "transcription", "data", "metadata")
DEFAULT_OUT = os.path.join(
    REPO_ROOT, "..", "fellowship-digital-ministry.github.io",
    "assets", "data", "sermons_catalog.json",
)


def build() -> list:
    entries = []
    for path in glob.glob(os.path.join(METADATA_DIR, "*_metadata.json")):
        try:
            m = json.load(open(path, encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        vid = m.get("video_id") or os.path.basename(path).replace("_metadata.json", "")
        notes = m.get("notes")
        entries.append({
            "video_id": vid,
            "title": m.get("title", f"Sermon {vid}"),
            "date": normalize_date(m.get("publish_date")),
            "duration": m.get("duration"),
            "url": f"https://www.youtube.com/watch?v={vid}",
            "description": (notes or {}).get("introduction", "") if notes else "",
            "notes": notes or None,
        })
    # Newest first. YYYYMMDD strings sort correctly lexicographically; blanks last.
    entries.sort(key=lambda e: e["date"], reverse=True)
    return entries


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=DEFAULT_OUT, help="output catalog JSON path")
    args = ap.parse_args()

    entries = build()
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=1)

    with_notes = sum(1 for e in entries if e["notes"])
    print(f"Wrote {len(entries)} sermons ({with_notes} with notes) to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
