#!/usr/bin/env python3
"""
YouTube channel monitoring script for sermon transcription pipeline.

This script checks for new videos on a YouTube channel, adds them to the video list CSV,
and processes them through the transcription pipeline.
"""
import argparse
import csv
import json
import logging
import os
import re
import requests
import subprocess
import sys
import time
import random
import hashlib
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from functools import lru_cache

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

# Cache directory for API responses
CACHE_DIR = ".api_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def cache_response(func):
    """Cache API responses to reduce quota usage and avoid throttling"""
    def wrapper(*args, **kwargs):
        # Create a cache key based on args
        key_parts = [str(arg) for arg in args]
        key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
        key_str = "|".join(key_parts)
        hash_key = hashlib.md5(key_str.encode()).hexdigest()
        cache_file = os.path.join(CACHE_DIR, f"{hash_key}.pkl")
        
        # Check if we have a fresh cache
        cache_valid_time = timedelta(hours=1)  # Cache for 1 hour
        if os.path.exists(cache_file):
            file_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))
            if file_age < cache_valid_time:
                try:
                    with open(cache_file, 'rb') as f:
                        return pickle.load(f)
                except Exception:
                    pass
        
        # Call the function and cache result
        result = func(*args, **kwargs)
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
        except Exception as e:
            logger.warning(f"Failed to cache response: {e}")
        
        return result
    
    return wrapper

def api_call_with_retry(func, *args, max_retries=3, **kwargs):
    """Make API calls with exponential backoff retry logic"""
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                logger.error(f"API call failed after {max_retries} attempts: {e}")
                return None
            
            # Calculate backoff time: 2^attempt * 0.5 second + random jitter
            backoff = (2 ** attempt) * 0.5 + random.uniform(0, 0.5)
            logger.warning(f"API call failed (attempt {attempt+1}/{max_retries}), "
                          f"retrying in {backoff:.2f}s: {e}")
            time.sleep(backoff)
    
    return None

def get_channel_id_from_handle(channel_handle: str) -> Optional[str]:
    """Convert a YouTube handle (@username) to a channel ID"""
    if channel_handle.startswith('@'):
        url = f"https://www.youtube.com/{channel_handle}"
    elif channel_handle.startswith('https://'):
        url = channel_handle
    else:
        url = f"https://www.youtube.com/@{channel_handle}"
    
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            match = re.search(r'"channelId":"([^"]+)"', response.text)
            if match:
                return match.group(1)
    except Exception as e:
        logger.error(f"Error extracting channel ID: {e}")
    
    return None

@cache_response
def fetch_playlist_items(api_key, playlist_id, max_results=5):
    """Fetch videos from a playlist using YouTube Data API"""
    url = "https://www.googleapis.com/youtube/v3/playlistItems"
    params = {
        "part": "snippet,contentDetails",
        "playlistId": playlist_id,
        "maxResults": max_results,
        "key": api_key
    }
    
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()

@cache_response
def fetch_video_details(api_key, video_id):
    """Fetch detailed information about a specific video"""
    url = "https://www.googleapis.com/youtube/v3/videos"
    params = {
        "part": "snippet,contentDetails,statistics",
        "id": video_id,
        "key": api_key
    }
    
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()

def fetch_channel_videos_api(channel_id: str, api_key: str, max_videos: int = 5) -> List[Dict]:
    """Fetch videos from a channel using the YouTube Data API"""
    logger.info(f"Fetching up to {max_videos} videos for channel {channel_id} using YouTube API")
    
    # Get uploads playlist ID (convert UC to UU)
    uploads_playlist_id = "UU" + channel_id[2:] if channel_id.startswith("UC") else None
    if not uploads_playlist_id:
        logger.error("Invalid channel ID format")
        return []
    
    # Fetch playlist items
    playlist_data = api_call_with_retry(fetch_playlist_items, api_key, uploads_playlist_id, max_videos)
    if not playlist_data:
        logger.error("Failed to fetch playlist items")
        return []
    
    videos = []
    for item in playlist_data.get('items', []):
        video_id = item.get('snippet', {}).get('resourceId', {}).get('videoId')
        if not video_id:
            continue
            
        logger.info(f"Found video: {video_id}")
        
        # Get detailed video information
        video_data = api_call_with_retry(fetch_video_details, api_key, video_id)
        if not video_data or not video_data.get('items'):
            logger.warning(f"No detailed data found for video {video_id}")
            continue
            
        video_info = video_data['items'][0]
        snippet = video_info.get('snippet', {})
        statistics = video_info.get('statistics', {})
        
        # Parse duration (in ISO 8601 format)
        duration_str = video_info.get('contentDetails', {}).get('duration', 'PT0S')
        duration_match = re.search(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', duration_str)
        hours = int(duration_match.group(1) or 0)
        minutes = int(duration_match.group(2) or 0)
        seconds = int(duration_match.group(3) or 0)
        duration = hours * 3600 + minutes * 60 + seconds
        
        # Format publish date
        publish_date = snippet.get('publishedAt', '')
        if publish_date:
            dt = datetime.strptime(publish_date, "%Y-%m-%dT%H:%M:%SZ")
            publish_date = dt.strftime('%Y%m%d')
        
        videos.append({
            "video_id": video_id,
            "title": snippet.get('title', ''),
            "description": snippet.get('description', ''),
            "publish_date": publish_date,
            "duration": duration,
            "view_count": int(statistics.get('viewCount', 0)),
            "like_count": int(statistics.get('likeCount', 0)),
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "thumbnail": snippet.get('thumbnails', {}).get('high', {}).get('url', '')
        })
        
        # Small delay to avoid rate limiting
        time.sleep(0.5)
    
    logger.info(f"Found {len(videos)} videos from YouTube API")
    return videos

def run_yt_dlp(args: List[str], cookies_file: Optional[str] = None, max_retries: int = 3) -> Tuple[bool, str]:
    """Run yt-dlp with given arguments and return output with retry logic"""
    command = ["yt-dlp"] + args
    
    # Add cookies if provided
    if cookies_file and os.path.exists(cookies_file):
        command.extend(["--cookies", cookies_file])
        logger.info(f"Using cookies from {cookies_file}")
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Running command (attempt {attempt+1}/{max_retries}): {' '.join(command)}")
            
            result = subprocess.run(
                command,
                text=True,
                capture_output=True,
                check=False
            )
            
            if result.returncode == 0:
                return True, result.stdout
            
            logger.warning(f"Command failed (attempt {attempt+1}/{max_retries}): {result.stderr}")
            
            # Wait before retrying (increasing delay)
            if attempt < max_retries - 1:
                sleep_time = (attempt + 1) * 5
                logger.info(f"Waiting {sleep_time}s before retrying...")
                time.sleep(sleep_time)
                
        except Exception as e:
            logger.error(f"Exception running command (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep((attempt + 1) * 5)
    
    return False, f"Failed after {max_retries} attempts"

def get_video_details_with_pytube(video_id: str) -> Optional[Dict]:
    """Get video details using PyTube"""
    try:
        from pytube import YouTube
    except ImportError:
        logger.error("PyTube not installed, cannot use PyTube fallback")
        return None
    
    url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        yt = YouTube(url)
        
        # Extract the publish date in YYYYMMDD format
        publish_date = ""
        if yt.publish_date:
            publish_date = yt.publish_date.strftime('%Y%m%d')
        
        return {
            "video_id": video_id,
            "title": yt.title or f"Unknown Title ({video_id})",
            "description": yt.description or "",
            "publish_date": publish_date,
            "duration": yt.length or 0,
            "view_count": yt.views or 0,
            "like_count": 0,  # PyTube doesn't get likes
            "url": url,
            "thumbnail": yt.thumbnail_url or ""
        }
    except Exception as e:
        logger.error(f"PyTube metadata error for {video_id}: {e}")
        return None
    
def download_audio_with_pytube(video_id: str, output_dir: str = "data/audio") -> bool:
    """Download audio using PyTube with robust error handling"""
    try:
        from pytube import YouTube
    except ImportError:
        logger.error("PyTube not installed, cannot download with PyTube")
        return False
    
    logger.info(f"Downloading audio for {video_id} using PyTube")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{video_id}.mp3")
    
    # If file already exists, skip download
    if os.path.exists(output_path):
        logger.info(f"Audio file already exists: {output_path}")
        return True
        
    url = f"https://www.youtube.com/watch?v={video_id}"
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            # Configure PyTube with timeout
            yt = YouTube(url)
            
            # Get highest quality audio stream
            audio_stream = yt.streams.filter(only_audio=True).order_by('abr').last()
            
            if not audio_stream:
                logger.warning(f"No audio stream found for {video_id}")
                return False
                
            # Download to temp file first
            temp_file = audio_stream.download(
                output_path=output_dir,
                filename=f"{video_id}.temp"
            )
            
            # Convert to MP3 using FFmpeg
            temp_path = os.path.join(output_dir, f"{video_id}.temp")
            subprocess.run([
                "ffmpeg", "-y", "-i", temp_path,
                "-b:a", "192k", output_path
            ], check=True, capture_output=True)
            
            # Remove temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
            logger.info(f"Successfully downloaded and converted {video_id}")
            return True
            
        except Exception as e:
            logger.error(f"PyTube download attempt {attempt+1} failed: {e}")
            time.sleep((attempt + 1) * 5)  # Exponential backoff
    
    logger.error(f"All download attempts failed for {video_id}")
    return False

def fetch_channel_videos(channel_url: str, cookies_file: Optional[str] = None, max_videos: int = 5) -> List[Dict]:
    """Fetch videos from the channel using yt-dlp with improved error handling"""
    # Arguments for yt-dlp to get channel metadata
    args = [
        "--dump-json",
        "--flat-playlist",
        "--playlist-end", str(max_videos),
        "--no-warnings",
        # Add user agent and headers to reduce blocking
        "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "--add-header", "Accept-Language:en-US,en;q=0.9",
        # Add delays to avoid rate limiting
        "--sleep-interval", "3",
        "--max-sleep-interval", "6",
        channel_url
    ]
    
    success, output = run_yt_dlp(args, cookies_file)
    
    if not success:
        logger.error("Failed to fetch channel videos")
        return []
    
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

def get_video_details(video_id: str, cookies_file: Optional[str] = None) -> Optional[Dict]:
    """Get detailed metadata for a specific video"""
    args = [
        "--dump-json",
        f"https://www.youtube.com/watch?v={video_id}",
        "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "--add-header", "Accept-Language:en-US,en;q=0.9",
        "--sleep-interval", "3", 
        "--max-sleep-interval", "6"
    ]
    
    success, output = run_yt_dlp(args, cookies_file)
    
    if success:
        try:
            video_info = json.loads(output)
            return video_info
        except json.JSONDecodeError:
            logger.error(f"Could not parse JSON for video {video_id}")
    
    # If getting metadata failed, try using PyTube as fallback
    logger.warning(f"yt-dlp metadata fetch failed for {video_id}, trying PyTube")
    video_info = get_video_details_with_pytube(video_id)
    
    if video_info:
        return video_info
    
    # If PyTube failed too, try direct audio download as last resort
    logger.warning(f"PyTube metadata fetch failed for {video_id}, trying direct download")
    
    # Create output directory if it doesn't exist
    os.makedirs("data/audio", exist_ok=True)
    
    if download_audio_with_pytube(video_id, "data/audio"):
        # Create minimal metadata
        return {
            "video_id": video_id,
            "title": f"Unknown Title ({video_id})",
            "description": "",
            "upload_date": datetime.now().strftime('%Y%m%d'),
            "duration": 0,
            "view_count": 0,
            "like_count": 0,
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "thumbnail": ""
        }
    
    return None

def load_video_list(csv_path: str) -> Tuple[Dict[str, Dict], List[str]]:
    """Load the video list CSV and return a dictionary of videos by ID"""
    videos = {}
    default_columns = [
        "video_id", "title", "description", "publish_date", 
        "duration", "view_count", "like_count", "url", "thumbnail",
        "processing_status", "processing_date", "transcript_path", "embeddings_status",
        "embeddings_date", "embeddings_count"
    ]
    
    if not os.path.exists(csv_path):
        logger.info(f"CSV file {csv_path} does not exist, will create a new one")
        return videos, default_columns
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            columns = reader.fieldnames or default_columns
            for row in reader:
                if 'video_id' in row and row['video_id']:
                    videos[row['video_id']] = row
        
        logger.info(f"Loaded {len(videos)} videos from {csv_path}")
        return videos, columns
    
    except Exception as e:
        logger.error(f"Error loading CSV file: {e}")
        return {}, default_columns

def save_video_list(videos: Dict[str, Dict], columns: List[str], csv_path: str):
    """Save videos dictionary to CSV file with backup"""
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
                # Ensure all columns exist in each row
                row = {col: video_data.get(col, "") for col in columns}
                writer.writerow(row)
        
        logger.info(f"Saved {len(videos)} videos to {csv_path}")
        
    except Exception as e:
        logger.error(f"Error saving to CSV file: {e}")

def check_for_new_videos_api(channel_id: str, api_key: str, csv_path: str, max_videos: int = 5) -> List[str]:
    """
    Check for new videos using the YouTube API and add them to the CSV.
    
    Args:
        channel_id: ID of the YouTube channel
        api_key: YouTube API key
        csv_path: Path to the video list CSV file
        max_videos: Maximum number of recent videos to check
        
    Returns:
        List of new video IDs
    """
    logger.info(f"Checking for new videos from channel ID: {channel_id} using YouTube API (max: {max_videos})")
    
    # Load existing videos
    existing_videos, columns = load_video_list(csv_path)
    
    # Fetch videos from channel
    channel_videos = fetch_channel_videos_api(channel_id, api_key, max_videos)
    
    if not channel_videos:
        logger.warning("No videos found or error fetching channel via API")
        return []
    
    # Check for new videos
    new_videos = []
    updated_videos = []
    
    for video in channel_videos:
        video_id = video.get('video_id')
        if not video_id:
            continue
            
        if video_id not in existing_videos:
            # This is a new video
            logger.info(f"Found new video: {video_id} - {video.get('title', 'Unknown Title')}")
            
            # Create CSV entry with data from API
            video_data = {
                "video_id": video_id,
                "title": video.get('title', 'Unknown Title'),
                "description": video.get('description', ''),
                "publish_date": video.get('publish_date', ''),
                "duration": video.get('duration', 0),
                "view_count": video.get('view_count', 0),
                "like_count": video.get('like_count', 0),
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "thumbnail": video.get('thumbnail', ''),
                "processing_status": "pending",
                "processing_date": "",
                "transcript_path": "",
                "embeddings_status": "pending",
                "embeddings_date": "",
                "embeddings_count": "0"
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
            existing_videos[video_id]["processing_status"] = "pending"
            updated_videos.append(video_id)
    
    # Save updated CSV
    save_video_list(existing_videos, columns, csv_path)
    
    logger.info(f"Found {len(new_videos)} new videos and {len(updated_videos)} videos to reprocess")
    
    # Return combined list of videos that need processing
    return new_videos + updated_videos

def check_for_new_videos(channel_url: str, csv_path: str, cookies_file: Optional[str] = None, max_videos: int = 5) -> List[str]:
    """
    Check for new videos on a YouTube channel and add them to the CSV.
    
    Args:
        channel_url: URL of the YouTube channel
        csv_path: Path to the video list CSV file
        cookies_file: Optional path to cookies file for authenticated requests
        max_videos: Maximum number of recent videos to check
        
    Returns:
        List of new video IDs
    """
    logger.info(f"Checking for new videos from: {channel_url} (max: {max_videos})")
    
    # Load existing videos
    existing_videos, columns = load_video_list(csv_path)
    
    # Fetch videos from channel
    channel_videos = fetch_channel_videos(channel_url, cookies_file, max_videos)
    
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
            detailed_info = get_video_details(video_id, cookies_file)
            
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
                "embeddings_status": "pending",
                "embeddings_date": "",
                "embeddings_count": "0"
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
    """Process newly discovered videos using the transcription pipeline."""
    logger.info(f"Running processing script for new videos")
    
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
    audio_dirs = ["data/audio", "audio", "temp_audio"]
    
    for audio_dir in audio_dirs:
        if not os.path.exists(audio_dir):
            continue
            
        try:
            count = 0
            for filename in os.listdir(audio_dir):
                if filename.endswith((".mp3", ".m4a", ".wav", ".ogg", ".mp4")):
                    file_path = os.path.join(audio_dir, filename)
                    os.remove(file_path)
                    count += 1
            
            if count > 0:
                logger.info(f"Cleaned up {count} audio files from {audio_dir}")
        except Exception as e:
            logger.error(f"Error cleaning up audio files in {audio_dir}: {e}")

def download_video_audio(video_id: str, output_dir: str = "data/audio", cookies_file: Optional[str] = None) -> bool:
    """
    Download just the audio from a YouTube video.
    First tries PyTube, falls back to yt-dlp if PyTube fails.
    
    Args:
        video_id: YouTube video ID
        output_dir: Directory to save the audio file
        cookies_file: Optional path to cookies file
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Downloading audio for video {video_id}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # First try with PyTube
    if download_audio_with_pytube(video_id, output_dir):
        return True
    
    # If PyTube failed, try with yt-dlp
    logger.info(f"PyTube download failed, trying with yt-dlp")
    
    # Arguments for yt-dlp to download audio
    args = [
        "-x",  # Extract audio
        "--audio-format", "mp3",  # Convert to MP3
        "--audio-quality", "0",  # Best quality
        "-o", f"{output_dir}/{video_id}.%(ext)s",  # Output filename
        "--no-warnings",
        # Add user agent and headers to reduce blocking
        "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "--add-header", "Accept-Language:en-US,en;q=0.9",
        # Add retries and delays
        "--retries", "5",
        "--sleep-interval", "3",
        "--max-sleep-interval", "6",
        f"https://www.youtube.com/watch?v={video_id}"
    ]
    
    success, output = run_yt_dlp(args, cookies_file, max_retries=3)
    
    return success

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Monitor YouTube channel for new videos")
    parser.add_argument("--channel", help="YouTube channel URL or handle")
    parser.add_argument("--output-dir", default="data", help="Output directory for data")
    parser.add_argument("--csv-file", help="Path to video list CSV (default: output-dir/video_list.csv)")
    parser.add_argument("--cookies", help="Path to YouTube cookies file")
    parser.add_argument("--process", action="store_true", help="Process new videos immediately")
    parser.add_argument("--cleanup", action="store_true", help="Delete audio files after processing")
    parser.add_argument("--max", type=int, default=5, help="How many recent videos to scan")
    
    # YouTube API related arguments
    parser.add_argument("--youtube-api", action="store_true", help="Use YouTube API instead of yt-dlp")
    parser.add_argument("--api-key", help="YouTube API key")
    parser.add_argument("--channel-id", help="YouTube channel ID (required for API mode)")
    
    args = parser.parse_args()
    
    # Set default CSV path if not provided
    if not args.csv_file:
        args.csv_file = os.path.join(args.output_dir, "video_list.csv")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.csv_file), exist_ok=True)
    
    # Decide which method to use
    new_videos = []
    if args.youtube_api:
        # Use YouTube API method
        if not args.api_key:
            logger.error("YouTube API key is required when using --youtube-api")
            return 1
        
        channel_id = args.channel_id
        if not channel_id and args.channel:
            # Try to extract channel ID from URL/handle
            channel_id = get_channel_id_from_handle(args.channel)
            if channel_id:
                logger.info(f"Extracted channel ID: {channel_id}")
            else:
                logger.error("Could not extract channel ID from provided channel URL")
                return 1
        
        if not channel_id:
            logger.error("Channel ID is required when using YouTube API")
            return 1
        
        logger.info(f"Using YouTube API to check for new videos (channel ID: {channel_id})")
        new_videos = check_for_new_videos_api(channel_id, args.api_key, args.csv_file, args.max)
        
    else:
        # Use yt-dlp method (legacy)
        if not args.channel:
            logger.error("Channel URL is required when not using YouTube API")
            return 1
        
        logger.info(f"Using yt-dlp to check for new videos (channel: {args.channel})")
        new_videos = check_for_new_videos(args.channel, args.csv_file, args.cookies, args.max)
    
    # Process if requested and new videos found
    if args.process and new_videos:
        # Before processing, ensure we have the audio files
        success_count = 0
        for video_id in new_videos:
            if download_video_audio(video_id, os.path.join(args.output_dir, "audio"), args.cookies):
                success_count += 1
                
        logger.info(f"Successfully downloaded audio for {success_count}/{len(new_videos)} videos")
        
        # Now process the videos
        if success_count > 0:
            process_new_videos(args.csv_file)
    
    # Clean up audio files if requested
    if args.cleanup:
        cleanup_audio_files()
    
    logger.info(f"Found {len(new_videos)} videos to process")
    return 0

if __name__ == "__main__":
    sys.exit(main())