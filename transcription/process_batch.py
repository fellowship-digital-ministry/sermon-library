"""
Main script for batch processing YouTube sermon videos with transcription tracking.

This script orchestrates the downloading and transcription of YouTube sermon videos
and maintains tracking information in a CSV file. It supports both single video
processing and batch processing with various options.

Features:
- Downloads audio from YouTube videos using yt-dlp
- Transcribes audio using OpenAI's Whisper API
- Tracks processing status and progress in a CSV file
- Supports proof-of-concept (POC) mode for testing with small batches
- Handles large audio files by splitting them into chunks
- Provides detailed logging and error handling

Usage examples:
  # Process a single YouTube URL:
  python process_batch.py --url "https://www.youtube.com/watch?v=VIDEO_ID"
  
  # Process 5 videos (POC mode default):
  python process_batch.py --csv data/video_list.csv
  
  # Process all pending/failed videos (full batch):
  python process_batch.py --csv data/video_list.csv --full
  
  # Process all videos regardless of status:
  python process_batch.py --csv data/video_list.csv --all --full
  
  # Force re-download of audio files:
  python process_batch.py --csv data/video_list.csv --force

CSV tracking:
  The script adds and updates these columns in the video list CSV:
  - processing_status: 'pending', 'in_progress', 'processed', or 'failed'
  - processing_date: Timestamp of last processing attempt
  - transcript_path: Path to the generated transcript file

Created for Fellowship Baptist Church Sermon Library
"""
import os
import logging
import argparse
import pandas as pd
from datetime import datetime
from tqdm import tqdm

import config
from utils import extract_video_id, save_transcript, save_metadata, get_video_url
from download_audio import download_batch, download_audio
from transcribe_audio import transcribe_batch, transcribe_audio

logger = logging.getLogger(__name__)

def update_video_status(csv_path, video_id, status, transcript_path=None):
    """
    Update the status of a video in the CSV file.
    
    Args:
        csv_path: Path to the CSV file
        video_id: YouTube video ID
        status: Processing status ('pending', 'in_progress', 'processed', 'failed')
        transcript_path: Path to the transcript file if processed
    """
    try:
        # Read CSV into DataFrame
        df = pd.read_csv(csv_path)
        
        # Find the row with matching video_id
        idx = df.index[df['video_id'] == video_id].tolist()
        
        if idx:
            # Update the row
            df.loc[idx[0], 'processing_status'] = status
            df.loc[idx[0], 'processing_date'] = datetime.now().isoformat()
            
            if transcript_path:
                df.loc[idx[0], 'transcript_path'] = transcript_path
        
        # Save updated CSV
        df.to_csv(csv_path, index=False)
        logger.info(f"Updated status for video {video_id} to {status}")
    except Exception as e:
        logger.error(f"Failed to update CSV status for {video_id}: {str(e)}")

def prepare_csv_with_status(csv_path):
    """
    Add status columns to the CSV if they don't exist.
    
    Args:
        csv_path: Path to the CSV file
    
    Returns:
        Path to the updated CSV file
    """
    try:
        if not os.path.exists(csv_path):
            logger.error(f"CSV file not found: {csv_path}")
            return csv_path
        
        # Read CSV
        df = pd.read_csv(csv_path)
        
        # Add status columns if they don't exist
        columns_to_add = {
            'processing_status': 'pending',
            'processing_date': '',
            'transcript_path': ''
        }
        
        modified = False
        for col, default_val in columns_to_add.items():
            if col not in df.columns:
                df[col] = default_val
                modified = True
        
        # Save if modified
        if modified:
            df.to_csv(csv_path, index=False)
            logger.info(f"Added status columns to {csv_path}")
        
        return csv_path
    except Exception as e:
        logger.error(f"Failed to prepare CSV: {str(e)}")
        return csv_path

def process_videos(csv_path=config.VIDEO_LIST_PATH, force_download=False, process_all=False):
    """
    Process videos from a CSV file with status tracking.
    
    Args:
        csv_path: Path to the CSV file
        force_download: Whether to download audio even if files already exist
        process_all: Whether to process all videos regardless of status
    
    Returns:
        Dictionary with processing statistics
    """
    start_time = datetime.now()
    
    # Prepare CSV with status columns
    csv_path = prepare_csv_with_status(csv_path)
    
    # Load videos from CSV
    try:
        df = pd.read_csv(csv_path)
        
        # Filter videos based on status if not processing all
        if not process_all:
            # Process videos that are pending or failed
            df = df[df['processing_status'].isin(['pending', 'failed', ''])]
        
        # Convert to list of dictionaries
        video_list = df.to_dict('records')
        
    except Exception as e:
        logger.error(f"Failed to load CSV: {str(e)}")
        return {"error": str(e)}
    
    # Apply POC limit if in POC mode
    if config.POC_MODE:
        original_count = len(video_list)
        video_list = video_list[:config.POC_LIMIT]
        logger.info(f"Running in POC mode: Processing {len(video_list)} of {original_count} videos")
    
    results = {
        "videos_processed": 0,
        "audio_files_downloaded": 0,
        "transcripts_created": 0,
        "failures": 0
    }
    
    # Process each video individually to track status
    for video in tqdm(video_list, desc="Processing videos"):
        video_id = video["video_id"]
        
        try:
            # Update status to in-progress
            update_video_status(csv_path, video_id, "in_progress")
            
            # Step 1: Download audio
            logger.info(f"Downloading audio for {video_id}")
            audio_file = download_audio(video_id, config.AUDIO_DIR, force_download)
            
            if not audio_file:
                update_video_status(csv_path, video_id, "failed")
                results["failures"] += 1
                continue
            
            results["audio_files_downloaded"] += 1
            
            # Step 2: Transcribe audio
            logger.info(f"Transcribing audio for {video_id}")
            transcript = transcribe_audio(audio_file)
            
            if not transcript:
                update_video_status(csv_path, video_id, "failed")
                results["failures"] += 1
                continue
            
            # Step 3: Save transcript
            transcript_file = save_transcript(video_id, transcript, config.TRANSCRIPT_DIR)
            results["transcripts_created"] += 1
            
            # Save metadata
            metadata = {
                "video_id": video_id,
                "title": video.get("title", ""),
                "publish_date": video.get("publish_date", ""),
                "audio_file": audio_file,
                "transcript_file": transcript_file,
                "duration": video.get("duration", 0),
                "transcription_timestamp": datetime.now().isoformat(),
                "model": config.WHISPER_MODEL,
            }
            save_metadata(video_id, metadata, config.METADATA_DIR)
            
            # Update status to processed
            update_video_status(csv_path, video_id, "processed", transcript_file)
            results["videos_processed"] += 1
            
        except Exception as e:
            logger.error(f"Error processing {video_id}: {str(e)}")
            update_video_status(csv_path, video_id, "failed")
            results["failures"] += 1
    
    # Calculate statistics
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    results.update({
        "processing_duration_seconds": duration,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "poc_mode": config.POC_MODE,
    })
    
    logger.info(f"Processing completed in {duration:.2f} seconds")
    logger.info(f"Videos processed: {results['videos_processed']}")
    logger.info(f"Audio files downloaded: {results['audio_files_downloaded']}")
    logger.info(f"Transcripts created: {results['transcripts_created']}")
    logger.info(f"Failures: {results['failures']}")
    
    return results

def process_single_url(youtube_url, force_download=False):
    """
    Process a single YouTube URL: extract ID, download audio, and transcribe.
    
    Args:
        youtube_url: YouTube URL to process
        force_download: Whether to download audio even if file already exists
    
    Returns:
        Transcript file path or None if processing failed
    """
    # Extract video ID from URL
    video_id = extract_video_id(youtube_url)
    if not video_id:
        logger.error(f"Could not extract video ID from URL: {youtube_url}")
        return None
    
    logger.info(f"Processing YouTube URL: {youtube_url} (ID: {video_id})")
    
    # Step 1: Download audio
    audio_file = download_audio(video_id, config.AUDIO_DIR, force_download)
    if not audio_file:
        logger.error(f"Failed to download audio for {youtube_url}")
        return None
    
    # Step 2: Transcribe audio
    transcript = transcribe_audio(audio_file)
    if not transcript:
        logger.error(f"Failed to transcribe audio for {youtube_url}")
        return None
    
    # Step 3: Save transcript
    transcript_file = save_transcript(video_id, transcript, config.TRANSCRIPT_DIR)
    
    # Save basic metadata
    metadata = {
        "video_id": video_id,
        "url": youtube_url,
        "audio_file": audio_file,
        "transcript_file": transcript_file,
        "transcription_timestamp": datetime.now().isoformat(),
        "model": config.WHISPER_MODEL,
    }
    save_metadata(video_id, metadata, config.METADATA_DIR)
    
    logger.info(f"Successfully processed {youtube_url}")
    logger.info(f"Transcript saved to: {transcript_file}")
    
    return transcript_file

if __name__ == "__main__":
    # Set up argument parser for command-line options
    parser = argparse.ArgumentParser(description="Process YouTube sermon videos for transcription")
    parser.add_argument("--force", action="store_true", help="Force download even if files exist")
    parser.add_argument("--full", action="store_true", help="Process full list (override POC mode)")
    parser.add_argument("--all", action="store_true", help="Process all videos regardless of status")
    parser.add_argument("--url", type=str, help="Process a single YouTube URL")
    parser.add_argument("--urls", type=str, help="File containing YouTube URLs (one per line)")
    parser.add_argument("--csv", type=str, default=config.VIDEO_LIST_PATH, 
                        help=f"Path to video list CSV (default: {config.VIDEO_LIST_PATH})")
    args = parser.parse_args()
    
    # Override POC mode if --full flag is provided
    if args.full:
        config.POC_MODE = False
        logger.info("Processing full video list (POC mode overridden)")
    
    # Process a single URL if provided
    if args.url:
        process_single_url(args.url, force_download=args.force)
    
    # Process multiple URLs from a file if provided
    elif args.urls:
        if not os.path.exists(args.urls):
            logger.error(f"URLs file not found: {args.urls}")
        else:
            with open(args.urls, 'r') as f:
                urls = [line.strip() for line in f if line.strip()]
            
            logger.info(f"Processing {len(urls)} URLs from file: {args.urls}")
            for url in urls:
                process_single_url(url, force_download=args.force)
    
    # Otherwise, process videos from the CSV file
    else:
        process_videos(csv_path=args.csv, force_download=args.force, process_all=args.all)