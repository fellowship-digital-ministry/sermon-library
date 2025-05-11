#!/usr/bin/env python3
"""
YouTube channel monitoring script for sermon transcription pipeline.

This script checks for new videos on a YouTube channel, adds them to the video list CSV,
and processes them through the transcription pipeline.

This version is designed to work without requiring cookies, which simplifies the process.
"""
import argparse
import csv
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("channel_monitor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_yt_dlp(args: List[str]) -> Tuple[bool, str]:
    """Run yt-dlp with given arguments and return output"""
    try:
        # Build the full command
        command = ["yt-dlp"] + args
        
        logger.info(f"Running command: {' '.join(command)}")
        
        # Run the command and capture output
        result = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False  # Don't raise an exception on non-zero return code
        )
        
        if result.returncode != 0:
            logger.error(f"yt-dlp command failed with code {result.returncode}")
            logger.error(f"Error output: {result.stderr}")
            return False, result.stderr
            
        return True, result.stdout
        
    except Exception as e:
        logger.error(f"Error running yt-dlp: {e}")
        return False, str(e)

def fetch_channel_videos(channel_url: str) -> List[Dict]:
    """Fetch videos from the channel using yt-dlp with retries"""
    # Arguments for yt-dlp to get channel metadata in JSON format
    args = [
        "--dump-json",  # Output video metadata as JSON
        "--flat-playlist",  # Don't download videos
        "--playlist-end", "30",  # Limit to last 30 videos (adjust as needed)
        "--no-warnings",  # Reduce noise in output
        channel_url  # Channel URL
    ]
    
    # Try up to 3 times
    for attempt in range(3):
        try:
            success, output = run_yt_dlp(args)
            
            if not success:
                logger.warning(f"Attempt {attempt+1}/3 failed, retrying in 10 seconds...")
                time.sleep(10)
                continue
            
            # Parse the JSON output (one JSON object per line)
            videos = []
            for line in output.strip().split('\n'):
                if line:
                    try:
                        video_info = json.loads(line)
                        videos.append(video_info)
                    except json.JSONDecodeError:
                        logger.warning(f"Could not parse JSON line: {line[:100]}...")
            
            logger.info(f"Found {len(videos)} videos in channel")
            return videos
            
        except Exception as e:
            logger.error(f"Error fetching channel videos (attempt {attempt+1}/3): {e}")
            if attempt < 2:  # Don't sleep after the last attempt
                time.sleep(10)  # Wait before retrying
    
    # If all attempts failed
    logger.error("Failed to fetch channel videos after 3 attempts")
    return []

def get_video_details(video_id: str) -> Optional[Dict]:
    """Get detailed metadata for a specific video with retries"""
    args = [
        "--dump-json",
        f"https://www.youtube.com/watch?v={video_id}"
    ]
    
    # Try up to 3 times
    for attempt in range(3):
        success, output = run_yt_dlp(args)
        
        if not success:
            logger.error(f"Failed to get details for video {video_id} (attempt {attempt+1}/3)")
            if attempt < 2:
                time.sleep(5)
                continue
            return None
        
        try:
            video_info = json.loads(output)
            return video_info
        except json.JSONDecodeError:
            logger.error(f"Could not parse JSON for video {video_id}")
            if attempt < 2:
                time.sleep(5)
                continue
            return None
    
    return None

def load_video_list(csv_path: str) -> Tuple[Dict[str, Dict], List[str]]:
    """Load the video list CSV and return a dictionary of videos by ID"""
    videos = {}
    columns = []
    
    if not os.path.exists(csv_path):
        logger.info(f"CSV file {csv_path} does not exist, will create a new one")
        return videos, [
            "video_id", "title", "description", "publish_date", 
            "duration", "view_count", "like_count", "url", "thumbnail",
            "processing_status", "processing_date", "transcript_path", "embeddings_status"
        ]
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            columns = reader.fieldnames or []
            for row in reader:
                if 'video_id' in row and row['video_id']:
                    videos[row['video_id']] = row
        
        logger.info(f"Loaded {len(videos)} videos from {csv_path}")
        return videos, columns
    
    except Exception as e:
        logger.error(f"Error loading CSV file: {e}")
        return {}, [
            "video_id", "title", "description", "publish_date", 
            "duration", "view_count", "like_count", "url", "thumbnail",
            "processing_status", "processing_date", "transcript_path", "embeddings_status"
        ]

def save_video_list(videos: Dict[str, Dict], columns: List[str], csv_path: str):
    """Save videos dictionary to CSV file"""
    try:
        # Make a backup of the existing file
        if os.path.exists(csv_path):
            backup_file = f"{csv_path}.bak"
            import shutil
            shutil.copy(csv_path, backup_file)
            logger.info(f"Created backup at {backup_file}")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for video_data in videos.values():
                writer.writerow(video_data)
        
        logger.info(f"Saved {len(videos)} videos to {csv_path}")
        
    except Exception as e:
        logger.error(f"Error saving to CSV file: {e}")

def check_for_new_videos(channel_url: str, csv_path: str) -> List[str]:
    """
    Check for new videos on a YouTube channel and add them to the CSV.
    
    Args:
        channel_url: URL of the YouTube channel
        csv_path: Path to the video list CSV file
        
    Returns:
        List of new video IDs
    """
    logger.info(f"Checking for new videos from: {channel_url}")
    
    # Load existing videos
    existing_videos, columns = load_video_list(csv_path)
    
    # Fetch videos from channel
    channel_videos = fetch_channel_videos(channel_url)
    
    if not channel_videos:
        logger.warning("No videos found or error fetching channel")
        return []
    
    # Check for new videos
    new_videos = []
    updated_videos = []
    
    for video in channel_videos:
        video_id = video.get('id')
        if not video_id:
            continue
            
        if video_id not in existing_videos:
            # This is a new video
            logger.info(f"Found new video: {video_id} - {video.get('title', 'Unknown Title')}")
            
            # Get detailed info
            detailed_info = get_video_details(video_id)
            
            if not detailed_info:
                logger.warning(f"Could not get detailed info for video {video_id}, skipping")
                continue
            
            # Create CSV entry
            video_data = {
                "video_id": video_id,
                "title": detailed_info.get('title', 'Unknown Title'),
                "description": detailed_info.get('description', ''),
                "publish_date": detailed_info.get('upload_date', ''),
                "duration": detailed_info.get('duration', 0),
                "view_count": detailed_info.get('view_count', 0),
                "like_count": detailed_info.get('like_count', 0),
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "thumbnail": detailed_info.get('thumbnail', ''),
                "processing_status": "pending",
                "processing_date": "",
                "transcript_path": "",
                "embeddings_status": "pending"
            }
            
            # Add any missing columns with empty values
            for col in columns:
                if col not in video_data:
                    video_data[col] = ""
            
            existing_videos[video_id] = video_data
            new_videos.append(video_id)
            
        elif existing_videos[video_id].get("processing_status") in ["failed", "pending", ""]:
            # This is a video that needs to be reprocessed
            logger.info(f"Video needs processing: {video_id} - {existing_videos[video_id].get('title', 'Unknown Title')}")
            updated_videos.append(video_id)
    
    # Save updated CSV
    save_video_list(existing_videos, columns, csv_path)
    
    logger.info(f"Found {len(new_videos)} new videos and {len(updated_videos)} videos to reprocess")
    
    # Return combined list of videos that need processing
    return new_videos + updated_videos

def process_new_videos(csv_path: str):
    """
    Process newly discovered videos using the transcription pipeline.
    
    Args:
        csv_path: Path to the video list CSV
    """
    logger.info(f"Running processing script for new videos")
    
    # Run the processing script
    try:
        cmd = ["python", "process_batch.py", "--csv", csv_path, "--full"]
        logger.info(f"Executing: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("Processing completed successfully")
            logger.info(result.stdout)
        else:
            logger.error(f"Processing failed: {result.stderr}")
    except Exception as e:
        logger.error(f"Error running processing script: {e}")

def cleanup_audio_files():
    """Delete all audio files in the audio directory after processing"""
    # Check for the different possible audio directories
    audio_dirs = ["data/audio", "audio", "temp_audio"]
    
    for audio_dir in audio_dirs:
        if not os.path.exists(audio_dir):
            continue
            
        try:
            count = 0
            for filename in os.listdir(audio_dir):
                if filename.endswith((".mp3", ".m4a", ".wav", ".ogg")):
                    file_path = os.path.join(audio_dir, filename)
                    os.remove(file_path)
                    count += 1
            
            if count > 0:
                logger.info(f"Cleaned up {count} audio files from {audio_dir}")
        except Exception as e:
            logger.error(f"Error cleaning up audio files in {audio_dir}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor YouTube channel for new videos")
    parser.add_argument("--channel", required=True, help="YouTube channel URL")
    parser.add_argument("--output-dir", default="data", help="Output directory for data")
    parser.add_argument("--csv-file", help="Path to video list CSV (default: output-dir/video_list.csv)")
    parser.add_argument("--cookies", help="Path to YouTube cookies file (not used in this version)")
    parser.add_argument("--process", action="store_true", help="Process new videos immediately")
    parser.add_argument("--cleanup", action="store_true", help="Delete audio files after processing")
    
    args = parser.parse_args()
    
    # Set default CSV path if not provided
    if not args.csv_file:
        args.csv_file = os.path.join(args.output_dir, "video_list.csv")
    
    # Check for new videos (ignoring cookies parameter)
    new_videos = check_for_new_videos(args.channel, args.csv_file)
    
    # Process if requested and new videos found
    if args.process and new_videos:
        process_new_videos(args.csv_file)
    
    # Clean up audio files if requested
    if args.cleanup:
        cleanup_audio_files()