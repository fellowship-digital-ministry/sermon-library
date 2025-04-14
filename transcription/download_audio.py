"""Script to download audio from YouTube videos.
"""
import os
import logging
import yt_dlp
from tqdm import tqdm

import config
from utils import get_video_url, get_audio_filename

logger = logging.getLogger(__name__)

def download_audio(video_id, output_dir=config.AUDIO_DIR, force_download=False):
    """
    Download audio from a YouTube video.
    
    Args:
        video_id: YouTube video ID
        output_dir: Directory to save the audio file
        force_download: Whether to download even if the file already exists
    
    Returns:
        Path to the downloaded audio file or None if download failed
    """
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = get_audio_filename(video_id, output_dir)
    
    # Skip if file already exists and force_download is False
    if os.path.exists(output_file) and not force_download:
        logger.info(f"Audio for {video_id} already exists at {output_file}")
        return output_file
    
    video_url = get_video_url(video_id)
    logger.info(f"Downloading audio from {video_url}")
    
    ydl_opts = {
        'format': config.YTDLP_FORMAT,
        'postprocessors': config.YTDLP_POSTPROCESSORS,
        'outtmpl': os.path.join(output_dir, f"{video_id}.%(ext)s"),
        'quiet': True,
        'no_warnings': True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        
        logger.info(f"Successfully downloaded audio for {video_id} to {output_file}")
        return output_file
    except Exception as e:
        logger.error(f"Failed to download audio for {video_id}: {str(e)}")
        return None

def download_batch(video_list, output_dir=config.AUDIO_DIR, force_download=False):
    """
    Download audio for a batch of videos.
    
    Args:
        video_list: List of dictionaries with video information
        output_dir: Directory to save audio files
        force_download: Whether to download even if files already exist
    
    Returns:
        Dictionary mapping video IDs to audio file paths
    """
    results = {}
    
    for video in tqdm(video_list, desc="Downloading audio"):
        video_id = video["video_id"]
        audio_file = download_audio(video_id, output_dir, force_download)
        
        if audio_file:
            results[video_id] = audio_file
    
    logger.info(f"Downloaded {len(results)} audio files")
    return results

if __name__ == "__main__":
    # This allows running this script directly for testing
    from utils import load_video_list
    
    # Load videos
    videos = load_video_list(config.VIDEO_LIST_PATH)
    
    # Apply POC limit if in POC mode
    if config.POC_MODE:
        videos = videos[:config.POC_LIMIT]
        logger.info(f"Running in POC mode with {len(videos)} videos")
    
    # Download audio
    download_batch(videos)