# Fellowship Baptist Church Sermon Library

A searchable digital library system for Pastor Mann's sermons at Fellowship Baptist Church in Oakton, Virginia.

## Project Overview

This project creates an AI-powered searchable archive of sermons that allows church members and visitors to:

- Search for specific topics across all sermons
- Ask questions and get answers based on sermon content
- Find timestamps and links to specific parts of sermon videos
- Access transcripts of all sermons

## Current Status

This is a community initiative project currently in development. The system will initially include:

- Automated transcription of YouTube sermon videos
- Searchable database of sermon content
- Simple web interface for queries

## Repository Structure

- `transcription/` - Python scripts for downloading and transcribing sermons
- `database/` - Vector database integration for sermon search
- `api/` - Backend API for querying sermon content
- `frontend/` - Jekyll-based web interface

## Getting Started

Documentation for setup and usage will be expanded as development progresses. The initial transcription system is now available in the `transcription/` directory.

## Audio Files

Raw audio files (.mp3) are not stored in the repository. Use the download script to retrieve them:

```bash
cd transcription
python download_audio.py --video-id <VIDEO_ID>
```

To download all videos listed in `transcription/data/video_list.csv` run:

```bash
python process_batch.py --csv transcription/data/video_list.csv
```


## Manual Transcription Workflow

You can run the transcription tools yourself using the scripts in `transcription/`. Always provide the YouTube channel ID to process (a channel handle will work once the script supports it). The default Fellowship Baptist Church channel ID is `UCek_LI7dZopFJEvwxDnovJg`.

```bash
cd transcription
python monitor_channel.py --channel-id UCek_LI7dZopFJEvwxDnovJg --process --cleanup
python process_batch.py --csv transcription/data/video_list.csv
```
### Windows Batch Workflow

For a simple end-to-end process on Windows, use the provided `process_sermons.bat` script. It downloads new videos from the channel, transcribes them, and generates Pinecone embeddings.

```cmd
process_sermons.bat
```

Pass a different channel ID as the first argument if needed.


## Contact

This is an unofficial community initiative. For more information about this project, please open an issue in this repository.

For official church information, please visit [Fellowship Baptist Church](https://www.fbcva.org/).

