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

def get_channel_id_from_handle(channel_handle: str) -> Optional[str]:
    """Convert a YouTube handle (@username) to a channel ID"""
    if channel_handle.startswith('@'):
        url = f"https://www.youtube.com/{channel_handle}"
    elif channel_handle.startswith('https://'):
        url = channel_handle
    else:
        url = f"https://www.youtube.com/@{channel_handle}"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            match = re.search(r'"channelId":"([^"]+)"', response.text)
            if match:
                return match.group(1)
    except Exception as e:
        logger.error(f"Error extracting channel ID: {e}")
    
    return None

def fetch_channel_videos_api(channel_id: str, api_key: str, max_videos: int = 5) -> List[Dict]:
    """Fetch videos from a channel using the YouTube Data API"""
    logger.info(f"Fetching up to {max_videos} videos for channel {channel_id} using YouTube API")
    
    # First, get the list of recent videos
    search_url = "https://www.googleapis.com/youtube/v3/search"
    search_params = {
        "part": "snippet",
        "channelId": channel_id,
        "maxResults": max_videos,
        "order": "date",
        "type": "video",
        "key": api_key
    }
    
    try:
        response = requests.get(search_url, params=search_params)
        response.raise_for_status()
        search_data = response.json()
        
        videos = []
        for item in search_data.get('items', []):
            video_id = item['id']['videoId']
            logger.info(f"Found video: {video_id}")
            
            # Get more detailed video information
            video_url = "https://www.googleapis.com/youtube/v3/videos"
            video_params = {
                "part": "snippet,contentDetails,statistics",
                "id": video_id,
                "key": api_key
            }
            
            video_response = requests.get(video_url, params=video_params)
            video_response.raise_for_status()
            video_data = video_response.json()
            
            if video_data.get('items'):
                video_info = video_data['items'][0]
                snippet = video_info['snippet']
                statistics = video_info.get('statistics', {})
                
                # Parse duration (in ISO 8601 format)
                duration_str = video_info.get('contentDetails', {}).get('duration', 'PT0S')
                # Convert ISO duration to seconds (simplified)
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
            
            # Add a small delay to avoid rate limiting
            time.sleep(0.5)
        
        logger.info(f"Found {len(videos)} videos from YouTube API")
        return videos
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching videos from YouTube API: {e}")
        return []

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
    
    # If getting metadata failed, try direct audio download
    logger.warning(f"Metadata fetch failed for {video_id}, trying direct audio download")
    
    # Create output directory if it doesn't exist
    os.makedirs("data/audio", exist_ok=True)
    
    audio_args = [
        "-x",
        "--audio-format", "mp3",
        "--audio-quality", "0",
        "-o", f"data/audio/{video_id}.%(ext)s",
        f"https://www.youtube.com/watch?v={video_id}"
    ]
    
    success, _ = run_yt_dlp(audio_args, cookies_file)
    if success:
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
    
    # If we couldn't get audio either, try pytube as a fallback
    try:
        from pytube import YouTube
        logger.info(f"Attempting pytube download for {video_id}")
        
        url = f"https://www.youtube.com/watch?v={video_id}"
        yt = YouTube(url)
        stream = yt.streams.filter(only_audio=True).first()
        
        if stream:
            os.makedirs("data/audio", exist_ok=True)
            stream.download(output_path="data/audio", filename=f"{video_id}.mp4")
            logger.info(f"Successfully downloaded {video_id} using pytube")
            
            return {
                "video_id": video_id,
                "title": yt.title or f"Unknown Title ({video_id})",
                "description": yt.description or "",
                "upload_date": datetime.now().strftime('%Y%m%d'),
                "duration": yt.length or 0,
                "view_count": yt.views or 0,
                "like_count": 0,
                "url": url,
                "thumbnail": yt.thumbnail_url or ""
            }
    except Exception as e:
        logger.error(f"Pytube fallback failed: {e}")
    
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
    Download just the audio from a YouTube video using yt-dlp.
    
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
    
    if not success:
        logger.error(f"Failed to download audio for video {video_id}")
        # Try fallback using pytube
        try:
            from pytube import YouTube
            logger.info(f"Attempting pytube download for {video_id}")
            
            url = f"https://www.youtube.com/watch?v={video_id}"
            yt = YouTube(url)
            stream = yt.streams.filter(only_audio=True).first()
            
            if stream:
                output_file = stream.download(output_path=output_dir, filename=f"{video_id}.mp4")
                logger.info(f"Successfully downloaded {video_id} using pytube to {output_file}")
                return True
            else:
                logger.error(f"No audio stream available for {video_id}")
                return False
        except Exception as e:
            logger.error(f"Pytube fallback failed: {e}")
            return False
    
    return True

def download_audio_from_youtube(video_id, output_dir="data/audio", cookies_file=None):
    """
    Download audio using either cookies or a direct API approach
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{video_id}.mp3")
    
    # Try using cookies if provided
    if cookies_file and os.path.exists(cookies_file):
        logger.info(f"Attempting download with cookies: {video_id}")
        success, _ = run_yt_dlp([
            "-x", 
            "--audio-format", "mp3",
            "--audio-quality", "0",
            "-o", f"{output_dir}/{video_id}.%(ext)s",
            f"https://www.youtube.com/watch?v={video_id}"
        ], cookies_file, max_retries=1)
        
        if success:
            logger.info(f"Successfully downloaded audio with cookies: {video_id}")
            return True
    
    # If cookies failed or not provided, try an alternative approach
    try:
        # Try using the Invidious API approach
        invidious_instances = [
            "https://invidious.snopyta.org",
            "https://inv.riverside.rocks",
            "https://invidio.xamh.de",
            "https://y.com.sb",
            "https://invidious.kavin.rocks"
        ]
        
        for instance in invidious_instances:
            try:
                logger.info(f"Trying Invidious instance {instance} for {video_id}")
                # First get video details from Invidious
                api_url = f"{instance}/api/v1/videos/{video_id}"
                response = requests.get(api_url, timeout=10)
                
                if response.status_code == 200:
                    video_data = response.json()
                    
                    # Find the audio stream
                    for fmt in video_data.get("adaptiveFormats", []):
                        if fmt.get("type", "").startswith("audio"):
                            audio_url = fmt.get("url")
                            if audio_url:
                                logger.info(f"Found audio URL via Invidious: {video_id}")
                                # Download the audio file
                                audio_response = requests.get(audio_url, stream=True, timeout=30)
                                if audio_response.status_code == 200:
                                    with open(output_file, 'wb') as f:
                                        for chunk in audio_response.iter_content(chunk_size=8192):
                                            f.write(chunk)
                                    logger.info(f"Successfully downloaded audio via Invidious: {video_id}")
                                    return True
                            break
            except Exception as e:
                logger.warning(f"Invidious instance {instance} failed: {e}")
                continue
        
        # Next option: Try the YouTube Music API approach
        logger.info(f"Trying YouTube Music API for {video_id}")
        try:
            from pytube import YouTube
            yt = YouTube(f"https://www.youtube.com/watch?v={video_id}")
            
            # Get audio-only stream
            stream = yt.streams.filter(only_audio=True).first()
            if stream:
                # Download to temporary file
                temp_file = stream.download(output_path=output_dir, filename=f"temp_{video_id}")
                
                # Convert to mp3 using ffmpeg if needed
                if not temp_file.endswith('.mp3'):
                    mp3_file = os.path.join(output_dir, f"{video_id}.mp3")
                    subprocess.run([
                        "ffmpeg", "-i", temp_file, "-vn", "-ab", "128k", 
                        "-ar", "44100", "-y", mp3_file
                    ], check=True, capture_output=True)
                    
                    # Remove temp file
                    os.remove(temp_file)
                    
                logger.info(f"Successfully downloaded audio with pytube: {video_id}")
                return True
        except Exception as e:
            logger.warning(f"YouTube Music API approach failed: {e}")
        
        # Final attempt: Try YouTube-DL with different IPs or proxy approach
        # This is a more advanced option that would require proxy rotation
            
        logger.error(f"All download methods failed for {video_id}")
        return False
        
    except Exception as e:
        logger.error(f"Error downloading audio for {video_id}: {e}")
        return False

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
        if args.youtube_api:
            # When using the API, we need to download the audio files separately
            for video_id in new_videos:
                download_video_audio(video_id, os.path.join(args.output_dir, "audio"), args.cookies)
        
        # Now process the videos
        process_new_videos(args.csv_file)
    
    # Clean up audio files if requested
    if args.cleanup:
        cleanup_audio_files()
    
    logger.info(f"Found {len(new_videos)} videos to process")
    return 0

if __name__ == "__main__":
    sys.exit(main())