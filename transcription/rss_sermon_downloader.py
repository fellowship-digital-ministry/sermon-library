#!/usr/bin/env python3
"""
RSS Sermon Downloader - Downloads sermons from church RSS feed with YouTube ID matching

This script monitors both a YouTube channel via RSS feed and a church's podcast RSS feed,
matches the entries using YouTube IDs, and downloads the sermons directly from the podcast feed.
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
from typing import Dict, List, Optional, Tuple, Any
import urllib.parse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("sermon_downloader.log"),
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

def get_church_rss_feed(feed_url: str) -> Optional[str]:
    """Get the RSS feed XML from the church's podcast feed"""
    logger.info(f"Fetching church RSS feed from: {feed_url}")
    
    try:
        response = requests.get(feed_url, timeout=30)
        response.raise_for_status()
        return response.text
    except Exception as e:
        logger.error(f"Error fetching church RSS feed: {e}")
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
        
        logger.info(f"Found {len(videos)} videos in YouTube feed")
        return videos
        
    except Exception as e:
        logger.error(f"Error parsing YouTube feed XML: {e}")
        return []

def parse_church_rss_feed(feed_xml: str) -> List[Dict[str, Any]]:
    """Parse church podcast RSS feed and extract sermon information with YouTube IDs"""
    if not feed_xml:
        return []
    
    try:
        root = ET.fromstring(feed_xml)
        namespace = {
            'atom': 'http://www.w3.org/2005/Atom',
            'itunes': 'http://www.itunes.com/dtds/podcast-1.0.dtd',
            'content': 'http://purl.org/rss/1.0/modules/content/'
        }
        
        sermons = []
        for item in root.findall('.//item'):
            # Get basic sermon info
            title = item.find('title').text if item.find('title') is not None else ""
            link = item.find('link').text if item.find('link') is not None else ""
            pub_date = item.find('pubDate').text if item.find('pubDate') is not None else ""
            
            # Get MP3 download URL
            mp3_url = ""
            enclosure = item.find('enclosure')
            if enclosure is not None:
                mp3_url = enclosure.get('url', '')
            
            # Get duration if available
            duration_text = ""
            duration_element = item.find('itunes:duration', namespace)
            if duration_element is not None and duration_element.text:
                duration_text = duration_element.text
            
            # Extract YouTube ID from subtitle or summary (if available)
            youtube_id = None
            youtube_url = ""
            
            subtitle = item.find('itunes:subtitle', namespace)
            if subtitle is not None and subtitle.text:
                youtube_url = subtitle.text.strip()
                youtube_id = extract_youtube_id(youtube_url)
            
            # If not found in subtitle, try summary
            if not youtube_id:
                summary = item.find('itunes:summary', namespace)
                if summary is not None and summary.text:
                    youtube_url = summary.text.strip()
                    youtube_id = extract_youtube_id(youtube_url)
            
            # Calculate duration in seconds if available
            duration_seconds = 0
            if duration_text:
                duration_seconds = parse_duration(duration_text)
            
            sermons.append({
                "title": title,
                "link": link,
                "pub_date": pub_date,
                "mp3_url": mp3_url,
                "youtube_id": youtube_id,
                "youtube_url": youtube_url,
                "duration": duration_seconds,
                "duration_text": duration_text
            })
        
        logger.info(f"Found {len(sermons)} sermons in church RSS feed")
        return sermons
        
    except Exception as e:
        logger.error(f"Error parsing church RSS feed: {e}")
        return []

def extract_youtube_id(url: str) -> Optional[str]:
    """Extract YouTube video ID from various YouTube URL formats"""
    if not url:
        return None
    
    # Try various patterns
    patterns = [
        r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/|youtube\.com/v/|youtube\.com/watch\?.*v=|youtube\.com/watch\?.*&v=)([^&\?#]+)',
        r'youtube\.com/shorts/([^&\?#]+)',
        r'youtu\.be/([^&\?#]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    # Special case for YouTube links with si parameter
    if '?si=' in url:
        parts = url.split('?si=')[0]
        for pattern in patterns:
            match = re.search(pattern, parts)
            if match:
                return match.group(1)
    
    return None

def parse_duration(duration_text: str) -> int:
    """Parse duration text (HH:MM:SS or MM:SS) to seconds"""
    if not duration_text:
        return 0
    
    try:
        parts = duration_text.split(':')
        if len(parts) == 3:  # HH:MM:SS
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        elif len(parts) == 2:  # MM:SS
            return int(parts[0]) * 60 + int(parts[1])
        else:
            # Try to parse as just seconds
            return int(duration_text)
    except ValueError:
        return 0

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

def download_from_rss_feed(video_id: str, mp3_url: str, output_dir: str = "data/audio") -> bool:
    """Download sermon audio directly from the church's RSS feed"""
    if not video_id or not mp3_url:
        logger.error(f"Missing video_id or mp3_url")
        return False
    
    logger.info(f"Downloading audio for {video_id} from church RSS feed: {mp3_url}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{video_id}.mp3")
    
    # If we already have the file, no need to download
    if os.path.exists(output_path):
        logger.info(f"Audio file already exists: {output_path}")
        return True
    
    try:
        # Download the MP3 file
        response = requests.get(mp3_url, stream=True, timeout=300)
        response.raise_for_status()
        
        # Get content length if available
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f:
            downloaded = 0
            last_log = time.time()
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Log progress every 5 seconds
                    if time.time() - last_log > 5:
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            logger.info(f"Downloaded {downloaded / (1024*1024):.2f} MB of {total_size / (1024*1024):.2f} MB ({percent:.1f}%)")
                        else:
                            logger.info(f"Downloaded {downloaded / (1024*1024):.2f} MB")
                        last_log = time.time()
        
        logger.info(f"Successfully downloaded {video_id} from church RSS feed")
        return True
    
    except Exception as e:
        logger.error(f"Error downloading from RSS feed: {e}")
        return False

def match_and_process_videos(youtube_videos: List[Dict], 
                             church_sermons: List[Dict], 
                             csv_path: str, 
                             output_dir: str = "data/audio") -> List[str]:
    """
    Match YouTube videos with church RSS sermons and download/process them
    
    Args:
        youtube_videos: List of YouTube videos from channel feed
        church_sermons: List of sermons from church RSS feed
        csv_path: Path to the video list CSV file
        output_dir: Directory to save audio files
        
    Returns:
        List of video IDs that were successfully processed
    """
    # Load existing videos from CSV
    existing_videos, columns = load_video_list(csv_path)
    
    # Create a lookup dictionary for church sermons by YouTube ID
    sermon_by_youtube_id = {}
    for sermon in church_sermons:
        if sermon.get('youtube_id'):
            sermon_by_youtube_id[sermon.get('youtube_id')] = sermon
    
    logger.info(f"Found {len(sermon_by_youtube_id)} sermons with YouTube IDs in church RSS feed")
    
    # Process YouTube videos and match with church RSS feed
    new_videos = []
    updated_videos = []
    processed_videos = []
    
    for video in youtube_videos:
        video_id = video.get('video_id')
        if not video_id:
            continue
        
        # Check if we have this video in the church RSS feed
        matching_sermon = sermon_by_youtube_id.get(video_id)
        
        # Update with info from the church RSS feed if available
        if matching_sermon:
            video['duration'] = matching_sermon.get('duration', 0)
            video['mp3_url'] = matching_sermon.get('mp3_url', '')
            logger.info(f"Matched YouTube video {video_id} with church sermon")
        
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
    
    # Combined list of videos that need processing
    videos_to_process = new_videos + updated_videos
    
    # Download and process videos
    for video_id in videos_to_process:
        # Get the video data
        video_data = existing_videos.get(video_id, {})
        title = video_data.get('title', f"Unknown Title ({video_id})")
        
        # Try to get the sermon from the church RSS feed
        matching_sermon = sermon_by_youtube_id.get(video_id)
        
        if matching_sermon and matching_sermon.get('mp3_url'):
            # Download directly from the church RSS feed
            mp3_url = matching_sermon.get('mp3_url', '')
            if download_from_rss_feed(video_id, mp3_url, output_dir):
                processed_videos.append(video_id)
                logger.info(f"Successfully downloaded audio for {video_id} from church RSS feed")
            else:
                logger.error(f"Failed to download audio for {video_id} from church RSS feed")
        else:
            logger.warning(f"No matching sermon found in church RSS feed for {video_id}")
    
    return processed_videos

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
    parser = argparse.ArgumentParser(description="Download sermons using both YouTube and church RSS feeds")
    parser.add_argument("--channel-id", help="YouTube channel ID")
    parser.add_argument("--channel", "--handle", dest="channel", help="Channel handle or URL (e.g. @name)")
    parser.add_argument("--church-rss", default="https://fbcministries.net/feed/podcast", 
                        help="URL to church's podcast RSS feed")
    parser.add_argument("--output-dir", default="data", help="Output directory for data")
    parser.add_argument("--csv-file", help="Path to video list CSV (default: output-dir/video_list.csv)")
    parser.add_argument("--process", action="store_true", help="Process new videos immediately", default=True)
    parser.add_argument("--cleanup", action="store_true", help="Delete audio files after processing", default=True)
    parser.add_argument("--max", type=int, default=10, help="How many recent videos to scan")
    
    args = parser.parse_args()
    
    # Set default CSV path if not provided
    if not args.csv_file:
        args.csv_file = os.path.join(args.output_dir, "video_list.csv")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.csv_file), exist_ok=True)
    
    # Determine channel ID
    channel_id = args.channel_id
    if args.channel:
        channel_id = channel_id_from_handle(args.channel)
        if not channel_id:
            logger.error("Unable to resolve channel ID from provided handle/URL")
            return 1
    if not channel_id:
        channel_id = "UCek_LI7dZopFJEvwxDnovJg"  # Default to Fellowship Baptist Church

    # Get videos from YouTube feed
    youtube_feed_xml = get_channel_feed(channel_id)
    youtube_videos = parse_youtube_feed(youtube_feed_xml)
    
    # Only keep the most recent videos
    youtube_videos = youtube_videos[:args.max]
    
    # Get sermons from church RSS feed
    church_feed_xml = get_church_rss_feed(args.church_rss)
    church_sermons = parse_church_rss_feed(church_feed_xml)
    
    # Match videos from YouTube feed with sermons from church RSS feed and process them
    processed_videos = match_and_process_videos(
        youtube_videos, 
        church_sermons, 
        args.csv_file, 
        os.path.join(args.output_dir, "audio")
    )
    
    # Now process the videos
    if args.process and processed_videos:
        process_new_videos(args.csv_file)
    
    # Clean up audio files if requested
    if args.cleanup:
        cleanup_audio_files()
    
    logger.info(f"Found {len(processed_videos)} videos to process")
    return 0

if __name__ == "__main__":
    sys.exit(main())