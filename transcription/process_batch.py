"""
Main script to run the sermon transcription pipeline.
"""
import os
import logging
import argparse
from datetime import datetime

import config
from utils import load_video_list, extract_video_id
from download_audio import download_batch, download_audio
from transcribe_audio import transcribe_batch, transcribe_audio

logger = logging.getLogger(__name__)

def process_videos(video_list=None, force_download=False):
    """
    Process a list of videos: download audio and transcribe.
    
    Args:
        video_list: List of dictionaries with video information, 
                   or None to load from config.VIDEO_LIST_PATH
        force_download: Whether to download audio even if files already exist
    
    Returns:
        Dictionary with processing statistics
    """
    start_time = datetime.now()
    
    # Load videos if not provided
    if video_list is None:
        video_list = load_video_list(config.VIDEO_LIST_PATH)
    
    # Apply POC limit if in POC mode
    if config.POC_MODE:
        original_count = len(video_list)
        video_list = video_list[:config.POC_LIMIT]
        logger.info(f"Running in POC mode: Processing {len(video_list)} of {original_count} videos")
    
    # Step 1: Download audio
    logger.info(f"Step 1: Downloading audio for {len(video_list)} videos")
    audio_files = download_batch(video_list, config.AUDIO_DIR, force_download)
    
    # Step 2: Transcribe audio
    logger.info(f"Step 2: Transcribing {len(audio_files)} audio files")
    transcript_files = transcribe_batch(audio_files, config.TRANSCRIPT_DIR, config.METADATA_DIR)
    
    # Calculate statistics
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    stats = {
        "videos_processed": len(video_list),
        "audio_files_downloaded": len(audio_files),
        "transcripts_created": len(transcript_files),
        "processing_duration_seconds": duration,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "poc_mode": config.POC_MODE,
    }
    
    logger.info(f"Processing completed in {duration:.2f} seconds")
    logger.info(f"Videos processed: {stats['videos_processed']}")
    logger.info(f"Audio files downloaded: {stats['audio_files_downloaded']}")
    logger.info(f"Transcripts created: {stats['transcripts_created']}")
    
    return stats

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
    from utils import save_transcript
    transcript_file = save_transcript(video_id, transcript, config.TRANSCRIPT_DIR)
    
    logger.info(f"Successfully processed {youtube_url}")
    logger.info(f"Transcript saved to: {transcript_file}")
    
    return transcript_file

if __name__ == "__main__":
    # Set up argument parser for command-line options
    parser = argparse.ArgumentParser(description="Process YouTube sermon videos for transcription")
    parser.add_argument("--force", action="store_true", help="Force download even if files exist")
    parser.add_argument("--full", action="store_true", help="Process full list (override POC mode)")
    parser.add_argument("--url", type=str, help="Process a single YouTube URL")
    parser.add_argument("--urls", type=str, help="File containing YouTube URLs (one per line)")
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
        process_videos(force_download=args.force)