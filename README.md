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
- `api/` - Backend API for querying sermon content

## Getting Started

Documentation for setup and usage will be expanded as development progresses. The initial transcription system is now available in the `transcription/` directory.

## Workflow

1. Run `rss_sermon_downloader.py` to update `data/video_list.csv` and download MP3 files.
2. Run `process_batch.py` to transcribe the audio and generate subtitle files.
3. Run `tools/transcript_to_embeddings.py` to push the transcripts to Pinecone.

### Windows Batch Workflow

The `process_sermons.bat` script automates the above steps on Windows:

```cmd
process_sermons.bat
```

Pass a different channel ID as the first argument if needed.

## Environment Variables

Set the following variables so the scripts can access OpenAI and Pinecone:

- `OPENAI_API_KEY`
- `PINECONE_API_KEY`
- `PINECONE_ENVIRONMENT` (defaults to `us-east-1`)
- `PINECONE_INDEX_NAME` (defaults to `sermon-embeddings`)


## Contact

This is an unofficial community initiative. For more information about this project, please open an issue in this repository.

For official church information, please visit [Fellowship Baptist Church](https://www.fbcva.org/).

