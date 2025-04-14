"""
YouTube channel monitoring script for sermon transcription pipeline.

This script checks for new videos on a YouTube channel, adds them to the video list CSV,
and processes them through the transcription pipeline.
"""
import yt_dlp
import pandas as pd
import os
import logging
from datetime import datetime
import subprocess
import time

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

def check_for_new_videos(channel_url, csv_path):
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
    try:
        df = pd.read_csv(csv_path)
        existing_ids = set(df['video_id'].tolist())
        logger.info(f"Found {len(existing_ids)} existing videos in CSV")
    except Exception as e:
        logger.error(f"Error loading CSV: {str(e)}")
        existing_ids = set()
    
    # Configure yt-dlp options to get recent videos
    ydl_opts = {
        'ignoreerrors': True,
        'extract_flat': True,
        'playlistend': 30,  # Check only recent videos
        'quiet': True,
        'no_warnings': True,
    }
    
    new_videos = []
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Get channel info
            channel_info = ydl.extract_info(channel_url, download=False)
            
            if 'entries' not in channel_info:
                logger.warning("No entries found in channel")
                return []
            
            # Check each video
            for entry in channel_info['entries']:
                if entry is None:
                    continue
                
                video_id = entry.get('id')
                
                # Skip if already in our database
                if video_id in existing_ids:
                    continue
                
                # Get detailed info for new video
                try:
                    detailed_opts = {
                        'ignoreerrors': True,
                        'quiet': True,
                    }
                    
                    with yt_dlp.YoutubeDL(detailed_opts) as detailed_ydl:
                        video_url = f"https://www.youtube.com/watch?v={video_id}"
                        video_info = detailed_ydl.extract_info(video_url, download=False)
                        
                        # Create video entry
                        new_video = {
                            'video_id': video_id,
                            'title': video_info.get('title', ''),
                            'description': video_info.get('description', ''),
                            'publish_date': video_info.get('upload_date', ''),
                            'duration': video_info.get('duration', 0),
                            'view_count': video_info.get('view_count', 0),
                            'url': video_url,
                            'processing_status': 'pending',
                            'processing_date': '',
                            'transcript_path': ''
                        }
                        
                        new_videos.append(new_video)
                        logger.info(f"Found new video: {video_id} - {new_video['title']}")
                        
                except Exception as e:
                    logger.error(f"Error getting details for video {video_id}: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error checking channel: {str(e)}")
    
    # Add new videos to CSV
    if new_videos:
        try:
            # If file exists, append; otherwise create
            if os.path.exists(csv_path):
                df_new = pd.DataFrame(new_videos)
                df = pd.concat([df, df_new], ignore_index=True)
            else:
                df = pd.DataFrame(new_videos)
            
            # Save updated CSV
            df.to_csv(csv_path, index=False)
            logger.info(f"Added {len(new_videos)} new videos to {csv_path}")
        except Exception as e:
            logger.error(f"Error updating CSV: {str(e)}")
    
    return [v['video_id'] for v in new_videos]

def process_new_videos(csv_path):
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
        logger.error(f"Error running processing script: {str(e)}")

def cleanup_audio_files():
    """Delete all MP3 files in the audio directory after processing"""
    audio_dir = "data/audio"
    try:
        count = 0
        for filename in os.listdir(audio_dir):
            if filename.endswith(".mp3"):
                file_path = os.path.join(audio_dir, filename)
                os.remove(file_path)
                count += 1
        
        logger.info(f"Cleaned up {count} audio files")
    except Exception as e:
        logger.error(f"Error cleaning up audio files: {str(e)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor YouTube channel for new videos")
    parser.add_argument("--channel", default="https://www.youtube.com/c/FellowshipBaptistChurch", 
                        help="YouTube channel URL")
    parser.add_argument("--csv", default="data/video_list.csv", 
                        help="Path to video list CSV")
    parser.add_argument("--process", action="store_true", 
                        help="Process new videos immediately")
    parser.add_argument("--cleanup", action="store_true", 
                        help="Delete audio files after processing")
    
    args = parser.parse_args()
    
    # Check for new videos
    new_videos = check_for_new_videos(args.channel, args.csv)
    
    # Process if requested and new videos found
    if args.process:
        process_new_videos(args.csv)
    
    # Clean up audio files if requested
    if args.cleanup:
        cleanup_audio_files()