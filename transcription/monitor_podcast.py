#!/usr/bin/env python3
"""Monitor a podcast RSS feed and process new episodes."""
import argparse
import csv
import os
import re
import logging
from datetime import datetime
from typing import Dict

import feedparser
import requests

from transcribe_audio import transcribe_audio
from utils import save_transcript, save_metadata

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("podcast_monitor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def sanitize_id(value: str) -> str:
    """Sanitize a string to be used as an episode ID."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", value)


def load_episode_list(csv_path: str) -> Dict[str, Dict]:
    episodes = {}
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                episodes[row["episode_id"]] = row
    return episodes


def save_episode_list(episodes: Dict[str, Dict], csv_path: str):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fieldnames = [
        "episode_id",
        "title",
        "publish_date",
        "url",
        "audio_url",
        "processing_status",
        "transcript_path",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ep in episodes.values():
            writer.writerow(ep)


def download_audio(audio_url: str, output_dir: str, episode_id: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    audio_path = os.path.join(output_dir, f"{episode_id}.mp3")
    if os.path.exists(audio_path):
        return audio_path
    logger.info(f"Downloading audio from {audio_url}")
    response = requests.get(audio_url, stream=True, timeout=60)
    response.raise_for_status()
    with open(audio_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    return audio_path


def process_episode(episode: Dict, audio_dir: str, transcript_dir: str, metadata_dir: str) -> str:
    audio_file = download_audio(episode["audio_url"], audio_dir, episode["episode_id"])
    transcript = transcribe_audio(audio_file)
    if not transcript:
        return "failed"
    transcript_file = save_transcript(episode["episode_id"], transcript, transcript_dir)
    metadata = {
        "episode_id": episode["episode_id"],
        "title": episode.get("title", ""),
        "publish_date": episode.get("publish_date", ""),
        "audio_file": audio_file,
        "transcript_file": transcript_file,
        "processed_at": datetime.now().isoformat(),
    }
    save_metadata(episode["episode_id"], metadata, metadata_dir)
    episode["transcript_path"] = transcript_file
    return "processed"


def check_feed(feed_url: str, csv_path: str, data_dir: str):
    episodes = load_episode_list(csv_path)
    feed = feedparser.parse(feed_url)
    new_count = 0

    for entry in feed.entries:
        ep_id_raw = entry.get("id") or entry.get("guid") or entry.get("link")
        ep_id = sanitize_id(ep_id_raw)
        if ep_id in episodes:
            continue
        audio_url = entry.get("enclosures", [{}])[0].get("href", "")
        publish_date = ""
        if entry.get("published"):
            try:
                dt = datetime(*entry.published_parsed[:6])
                publish_date = dt.strftime("%Y%m%d")
            except Exception:
                publish_date = entry.get("published", "")
        episodes[ep_id] = {
            "episode_id": ep_id,
            "title": entry.get("title", f"Episode {ep_id}"),
            "publish_date": publish_date,
            "url": entry.get("link", ""),
            "audio_url": audio_url,
            "processing_status": "pending",
            "transcript_path": "",
        }
        new_count += 1
    if new_count:
        logger.info(f"Found {new_count} new episodes")
        save_episode_list(episodes, csv_path)
    else:
        logger.info("No new episodes found")
    return episodes


def main():
    parser = argparse.ArgumentParser(description="Monitor podcast RSS feed")
    parser.add_argument("--feed-url", required=True, help="Podcast RSS feed URL")
    parser.add_argument("--output-dir", default="transcription/data", help="Data directory")
    parser.add_argument("--process", action="store_true", help="Process new episodes")
    args = parser.parse_args()

    csv_path = os.path.join(args.output_dir, "podcast_list.csv")
    audio_dir = os.path.join(args.output_dir, "audio")
    transcript_dir = os.path.join(args.output_dir, "transcripts")
    metadata_dir = os.path.join(args.output_dir, "metadata")

    episodes = check_feed(args.feed_url, csv_path, args.output_dir)
    if args.process:
        for ep in episodes.values():
            if ep.get("processing_status") != "pending":
                continue
            status = process_episode(ep, audio_dir, transcript_dir, metadata_dir)
            ep["processing_status"] = status
        save_episode_list(episodes, csv_path)

    logger.info("Done")


if __name__ == "__main__":
    main()
