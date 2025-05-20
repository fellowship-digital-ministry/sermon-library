#!/usr/bin/env python3
"""Command-line tool to run the full sermon processing pipeline.

This script orchestrates downloading new sermons from the RSS feed,
processing audio/transcripts, and (optionally) generating embeddings.
"""

import argparse
import sys
from pathlib import Path

# Import pipeline modules
from transcription import rss_sermon_downloader, process_batch
from tools import transcript_to_embeddings


def run_rss_downloader(args) -> int:
    """Run rss_sermon_downloader.main() with the provided arguments.

    The original script always processes and cleans up files. We
    temporarily disable those behaviours so this wrapper can control
    subsequent steps itself.
    """
    # Monkey patch processing/cleanup so we only download/update the CSV
    rss_sermon_downloader.process_new_videos = lambda *_a, **_k: None
    rss_sermon_downloader.cleanup_audio_files = lambda: None

    argv = ["rss_sermon_downloader"]
    if args.channel_id:
        argv += ["--channel-id", args.channel_id]
    if args.channel:
        argv += ["--channel", args.channel]
    if args.church_rss:
        argv += ["--church-rss", args.church_rss]
    if args.output_dir:
        argv += ["--output-dir", args.output_dir]
    if args.csv_file:
        argv += ["--csv-file", args.csv_file]
    if args.max:
        argv += ["--max", str(args.max)]

    prev_argv = sys.argv
    sys.argv = argv
    try:
        return rss_sermon_downloader.main()
    finally:
        sys.argv = prev_argv


def run_process_batch(args):
    """Run process_batch.process_videos with arguments from CLI."""
    process_batch.process_videos(
        csv_path=args.csv_file,
        force_download=args.force,
        process_all=args.process_all,
        cookies_file=args.cookies,
        delay=args.delay,
    )


def run_embeddings(args):
    """Invoke transcript_to_embeddings.main() if requested."""
    argv = ["transcript_to_embeddings"]
    if args.csv_file:
        argv += ["--video_list_csv", args.csv_file]
    transcript_dir = Path(args.csv_file).parent / "transcripts"
    argv += ["--transcript_dir", str(transcript_dir)]
    prev_argv = sys.argv
    sys.argv = argv
    try:
        transcript_to_embeddings.main()
    finally:
        sys.argv = prev_argv


def main():
    parser = argparse.ArgumentParser(description="Run sermon processing pipeline")
    parser.add_argument("--channel-id", help="YouTube channel ID")
    parser.add_argument("--channel", help="Channel handle or URL")
    parser.add_argument(
        "--church-rss",
        default="https://fbcministries.net/feed/podcast",
        help="Church podcast RSS feed URL",
    )
    parser.add_argument("--output-dir", default="data", help="Output data directory")
    parser.add_argument(
        "--csv-file",
        help="Path to video_list.csv (default: output_dir/video_list.csv)",
    )
    parser.add_argument("--max", type=int, default=10, help="Number of recent videos to scan")

    parser.add_argument("--force", action="store_true", help="Force re-download audio")
    parser.add_argument(
        "--process-all",
        action="store_true",
        help="Process all videos regardless of status",
    )
    parser.add_argument("--cookies", help="Path to YouTube cookies file")
    parser.add_argument("--delay", type=float, default=0, help="Delay between downloads")
    parser.add_argument(
        "--embeddings",
        action="store_true",
        help="Generate embeddings after transcription",
    )

    args = parser.parse_args()

    # Determine CSV path
    if not args.csv_file:
        args.csv_file = str(Path(args.output_dir) / "video_list.csv")

    # Step 1: Download and update video list from RSS feeds
    run_rss_downloader(args)

    # Step 2: Transcribe downloaded videos
    run_process_batch(args)

    # Step 3: Optionally generate embeddings
    if args.embeddings:
        run_embeddings(args)


if __name__ == "__main__":
    main()
