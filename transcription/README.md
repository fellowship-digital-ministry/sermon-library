# Transcription Scripts

This folder contains the tools for downloading sermons, transcribing them and generating subtitles.

## Key Scripts

- `rss_sermon_downloader.py` – downloads new MP3 files and updates `data/video_list.csv` from the sermon RSS feed.
- `process_batch.py` – transcribes the downloaded audio files and creates VTT/SRT subtitle files.
- `json_to_srt.py` – converts JSON transcripts into subtitle formats.
- `monitor_channel.py` – legacy helper for monitoring a YouTube channel.

The workflow in the project root describes how these scripts fit together.
