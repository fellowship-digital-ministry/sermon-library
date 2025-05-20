#!/usr/bin/env python3
"""
PubSub Monitor - Alternative approach to fetch YouTube videos using RSS feeds

This script monitors a YouTube channel via RSS feed, which is less likely to be flagged
as a bot compared to direct YouTube API or scraping methods. Provide either
``--channel-id`` or ``--channel``/``--handle`` to specify which channel to watch.
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
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pubsub_monitor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def channel_id_from_handle(channel: str) -> Optional[str]:
    """Resolve a YouTube channel handle or URL to a channel ID."""
    if not channel:
        return None

    url = channel.strip()
    if url.startswith('@'):
        url = f"https://www.youtube.com/{url}"
    elif not url.startswith('http'):
        url = f"https://www.youtube.com/{url.lstrip('/')}"

    # Try pytube first
    try:
        from pytube import Channel
        ch = Channel(url)
        if ch.channel_id:
            return ch.channel_id
    except Exception as e:
        logger.warning(f"pytube failed to get channel id: {e}")

    # Fallback to scraping
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        match = re.search(r'"channelId":"(UC[^"]+)"', response.text)
        if match:
            return match.group(1)
    except Exception as e:
        logger.error(f"Failed to scrape channel id: {e}")

    return None

def get_channel_feed(channel_id: str) -> Optional[str]:
    """Get the RSS feed XML for a YouTube channel"""
    feed_url = f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
    logger.info(f"Fetching RSS feed from: {feed_url}")
    
    try:
        response = requests.get(feed_url, timeout=30)
        response.raise_for_status()
        return response.text
    except Exception as e:
        logger.error(f"Error fetching channel feed: {e}")
        return None

def parse_youtube_feed(feed_xml: str) -> List[Dict]:
    """Parse YouTube RSS feed XML and extract video information"""
    if not feed_xml:
        return []
    
    try:
        root = ET.fromstring(feed_xml)
        namespace = {'atom': 'http://www.w3.org/2005/Atom', 
                    'media': 'http://search.yahoo.com/mrss/'}
        
        videos = []
        for entry in root.findall('.//atom:entry', namespace):
            # Get video ID from link
            link = entry.find('atom:link', namespace).get('href')
            video_id = link.split('v=')[-1] if 'v=' in link else link.split('/')[-1]
            
            # Get basic metadata
            title = entry.find('atom:title', namespace).text
            description = entry.find('media:group/media:description', namespace).text
            published = entry.find('atom:published', namespace).text
            
            # Convert published date to YYYYMMDD format
            if published:
                dt = datetime.fromisoformat(published.replace('Z', '+00:00'))
                publish_date = dt.strftime('%Y%m%d')
            else:
                publish_date = ''
            
            # Get thumbnail URL if available
            thumbnail = ''
            thumbnail_element = entry.find('media:group/media:thumbnail', namespace)
            if thumbnail_element is not None:
                thumbnail = thumbnail_element.get('url', '')
            
            videos.append({
                "video_id": video_id,
                "title": title or f"Unknown Title ({video_id})",
                "description": description or "",
                "publish_date": publish_date,
                "duration": 0,  # Not available in RSS feed
                "view_count": 0,  # Not available in RSS feed
                "like_count": 0,  # Not available in RSS feed
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "thumbnail": thumbnail
            })
        
        logger.info(f"Found {len(videos)} videos in feed")
        return videos
        
    except Exception as e:
        logger.error(f"Error parsing feed XML: {e}")
        return []

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

def download_audio_with_youtube_dl(video_id: str, output_dir: str = "data/audio") -> bool:
    """
    Download audio using youtube-dl (pure Python version, should be less detectable)
    This uses a different Python library that may have better success rates
    """
    try:
        # We'll use this as a last resort after yt-dlp and pytube have failed
        from youtube_dl import YoutubeDL
    except ImportError:
        try:
            # Try to install it dynamically if not available
            subprocess.check_call([sys.executable, "-m", "pip", "install", "youtube-dl"])
            from youtube_dl import YoutubeDL
        except:
            logger.error("Failed to install youtube-dl")
            return False
    
    logger.info(f"Downloading audio for {video_id} using youtube-dl")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{video_id}.%(ext)s")
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': output_path,
        'quiet': True,
        'no_warnings': True,
        'ignoreerrors': False,
        'nocheckcertificate': True,
        'retries': 5,
        'socket_timeout': 30,
    }
    
    try:
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([f'https://www.youtube.com/watch?v={video_id}'])
        
        # Check if file exists
        mp3_path = os.path.join(output_dir, f"{video_id}.mp3")
        if os.path.exists(mp3_path):
            logger.info(f"Successfully downloaded {video_id} with youtube-dl")
            return True
    except Exception as e:
        logger.error(f"youtube-dl download failed: {e}")
    
    return False

def download_from_alternative_source(video_id: str, title: str, output_dir: str = "data/audio") -> bool:
    """
    Try to download sermon audio from alternative sources like sermon aggregators
    
    Args:
        video_id: YouTube video ID (for naming the output file)
        title: The sermon title to search for
        output_dir: Directory to save the audio file
        
    Returns:
        True if successful, False otherwise
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{video_id}.mp3")
    
    # Try to find this sermon on SermonAudio.com
    search_terms = re.sub(r'[^\w\s]', '', title).strip().split()
    if len(search_terms) > 3:
        # Use just the first few key terms
        search_terms = search_terms[:3]
    
    search_query = "+".join(search_terms)
    
    try:
        # Try to search SermonAudio.com or a similar site
        # (This is a simplified example - you would need to implement the actual search)
        logger.info(f"Searching for sermon using terms: {search_query}")
        
        # If implementing this for real, you would:
        # 1. Search a sermon aggregator site
        # 2. Parse the results
        # 3. Download a matching sermon
        # 4. Save to the output_path
        
        # For now, we'll just return False
        return False
        
    except Exception as e:
        logger.error(f"Alternative source download failed: {e}")
        return False

def check_for_new_videos_feed(channel_id: str, csv_path: str, max_videos: int = 5) -> List[str]:
    """
    Check for new videos via RSS feed and add them to the CSV.
    
    Args:
        channel_id: ID of the YouTube channel
        csv_path: Path to the video list CSV file
        max_videos: Maximum number of recent videos to check
        
    Returns:
        List of new video IDs
    """
    logger.info(f"Checking for new videos from channel ID: {channel_id} via RSS feed")
    
    # Load existing videos
    existing_videos, columns = load_video_list(csv_path)
    
    # Fetch videos from feed
    feed_xml = get_channel_feed(channel_id)
    channel_videos = parse_youtube_feed(feed_xml)
    
    if not channel_videos:
        logger.warning("No videos found or error fetching channel via feed")
        return []
    
    # Only keep the most recent videos
    channel_videos = channel_videos[:max_videos]
    
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
            
            # Create CSV entry with data from feed
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

def download_video_audio(video_id: str, title: str, output_dir: str = "data/audio", cookies_file: Optional[str] = None) -> bool:
    """Download audio using multiple methods"""
    logger.info(f"Attempting to download audio for {video_id}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{video_id}.mp3")
    
    # If we already have the file, no need to download
    if os.path.exists(output_path):
        logger.info(f"Audio file already exists: {output_path}")
        return True
    
    # Try multiple methods
    
    # First try using subprocess to run yt-dlp with cookies
    if cookies_file and os.path.exists(cookies_file):
        try:
            logger.info(f"Trying yt-dlp with cookies")
            cmd = [
                "yt-dlp",
                "-x",
                "--audio-format", "mp3",
                "--audio-quality", "0",
                "-o", f"{output_dir}/{video_id}.%(ext)s",
                "--cookies", cookies_file,
                "--user-agent", "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "--sleep-interval", "15", 
                "--max-sleep-interval", "30",
                f"https://www.youtube.com/watch?v={video_id}"
            ]
            
            logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Successfully downloaded {video_id} with yt-dlp and cookies")
                return True
            else:
                logger.warning(f"yt-dlp with cookies failed: {result.stderr}")
        except Exception as e:
            logger.error(f"Error running yt-dlp: {e}")
    
    # Next try using youtube-dl (alternative Python library)
    try:
        if download_audio_with_youtube_dl(video_id, output_dir):
            return True
    except Exception as e:
        logger.error(f"youtube-dl approach failed: {e}")
    
    # Try pytube
    try:
        from pytube import YouTube
        logger.info(f"Trying pytube")
        
        yt = YouTube(f"https://www.youtube.com/watch?v={video_id}")
        audio_stream = yt.streams.filter(only_audio=True).first()
        
        if audio_stream:
            temp_file = audio_stream.download(output_path=output_dir, filename=f"{video_id}.temp")
            
            # Convert to MP3 using FFmpeg
            temp_path = os.path.join(output_dir, f"{video_id}.temp")
            subprocess.run([
                "ffmpeg", "-y", "-i", temp_path,
                "-b:a", "192k", output_path
            ], check=True, capture_output=True)
            
            # Remove temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
            logger.info(f"Successfully downloaded {video_id} with pytube")
            return True
    except Exception as e:
        logger.error(f"Pytube approach failed: {e}")
    
    # As a last resort, try alternative sermon sources
    if download_from_alternative_source(video_id, title, output_dir):
        return True
    
    # All methods failed
    logger.error(f"All download methods failed for {video_id}")
    return False

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

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Monitor YouTube channel via RSS feed")
    parser.add_argument("--channel-id", help="YouTube channel ID")
    parser.add_argument("--channel", "--handle", dest="channel", help="Channel handle or URL (e.g. @name)")
    parser.add_argument("--output-dir", default="data", help="Output directory for data")
    parser.add_argument("--csv-file", help="Path to video list CSV (default: output-dir/video_list.csv)")
    parser.add_argument("--cookies", help="Path to YouTube cookies file")
    parser.add_argument("--process", action="store_true", help="Process new videos immediately", default=True)
    parser.add_argument("--cleanup", action="store_true", help="Delete audio files after processing", default=True)
    parser.add_argument("--max", type=int, default=5, help="How many recent videos to scan")
    
    args = parser.parse_args()
    
    # Set default CSV path if not provided
    if not args.csv_file:
        args.csv_file = os.path.join(args.output_dir, "video_list.csv")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.csv_file), exist_ok=True)
    
    # Check for cookies file in current directory if not provided
    if not args.cookies:
        potential_cookie_files = ["youtube_cookies.txt", "../youtube_cookies.txt"]
        for cookie_file in potential_cookie_files:
            if os.path.exists(cookie_file):
                args.cookies = cookie_file
                logger.info(f"Using found cookies file: {args.cookies}")
                break
    
    # Determine channel ID
    channel_id = args.channel_id
    if args.channel:
        channel_id = channel_id_from_handle(args.channel)
        if not channel_id:
            logger.error("Unable to resolve channel ID from provided handle/URL")
            return 1
    if not channel_id:
        channel_id = "UCek_LI7dZopFJEvwxDnovJg"

    # Get videos from feed
    new_videos = check_for_new_videos_feed(channel_id, args.csv_file, args.max)
    
    # Load the video list to get titles
    existing_videos, _ = load_video_list(args.csv_file)
    
    # Process if requested and new videos found
    if args.process and new_videos:
        # Download audio for each video
        success_count = 0
        for video_id in new_videos:
            title = existing_videos.get(video_id, {}).get('title', f"Unknown Title ({video_id})")
            if download_video_audio(video_id, title, os.path.join(args.output_dir, "audio"), args.cookies):
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