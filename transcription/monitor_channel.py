import os
import sys
import json
import logging
import argparse
import subprocess
import pandas as pd
from datetime import datetime, timedelta
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SermonMonitor:
    def __init__(self, channel_url, cookies_path=None):
        self.channel_url = channel_url
        self.cookies_path = cookies_path
        self.data_dir = "data"
        self.video_list_file = os.path.join(self.data_dir, "video_list.csv")
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Load existing video list
        self.load_video_list()
    
    def load_video_list(self):
        """Load existing video list from CSV"""
        if os.path.exists(self.video_list_file):
            try:
                self.df_videos = pd.read_csv(self.video_list_file)
                logger.info(f"Found {len(self.df_videos)} existing videos in CSV")
            except pd.errors.EmptyDataError:
                logger.warning("CSV file is empty, creating new DataFrame")
                self.df_videos = pd.DataFrame(columns=['video_id', 'title', 'upload_date', 'processed', 'last_check'])
        else:
            logger.info("No existing video list found, creating new DataFrame")
            self.df_videos = pd.DataFrame(columns=['video_id', 'title', 'upload_date', 'processed', 'last_check'])
    
    def get_channel_videos(self, max_videos=50):
        """Get recent videos from YouTube channel using yt-dlp"""
        cmd = [
            'yt-dlp',
            '--dump-json',
            '--flat-playlist',
            '--playlist-items', f'1:{max_videos}',
            '--no-download'
        ]
        
        # Add cookies if available
        if self.cookies_path and os.path.exists(self.cookies_path):
            cmd.extend(['--cookies', self.cookies_path])
            logger.info(f"Using cookies from: {self.cookies_path}")
        else:
            logger.warning("No cookies file provided or file not found")
        
        cmd.append(self.channel_url)
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Error fetching channel videos: {result.stderr}")
                return []
            
            videos = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    try:
                        video_data = json.loads(line)
                        videos.append(video_data)
                    except json.JSONDecodeError:
                        continue
            
            logger.info(f"Found {len(videos)} videos from channel")
            return videos
            
        except Exception as e:
            logger.error(f"Error getting channel videos: {e}")
            return []
    
    def get_video_details(self, video_id):
        """Get detailed information about a specific video"""
        cmd = ['yt-dlp', '--dump-json', '--no-download']
        
        # Add cookies if available
        if self.cookies_path and os.path.exists(self.cookies_path):
            cmd.extend(['--cookies', self.cookies_path])
        
        cmd.append(f'https://www.youtube.com/watch?v={video_id}')
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                return json.loads(result.stdout)
            else:
                logger.error(f"Error getting details for video {video_id}: {result.stderr}")
                # Check if it's an authentication error and suggest solution
                if "Sign in to confirm you're not a bot" in result.stderr:
                    logger.error("Authentication required. Please ensure cookies are properly configured.")
                return None
                
        except Exception as e:
            logger.error(f"Error getting details for video {video_id}: {e}")
            return None
    
    def check_for_new_videos(self):
        """Check for new videos and update the CSV"""
        logger.info(f"Checking for new videos from: {self.channel_url}")
        
        # Get recent videos from channel
        recent_videos = self.get_channel_videos()
        
        if not recent_videos:
            logger.warning("No videos found from channel")
            return
        
        # Track new videos
        new_videos = []
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        for video in recent_videos:
            video_id = video.get('id')
            if not video_id:
                continue
            
            # Check if video already exists in our list
            existing_video = self.df_videos[self.df_videos['video_id'] == video_id]
            
            if existing_video.empty:
                # Get detailed information for new video
                video_details = self.get_video_details(video_id)
                
                if video_details:
                    new_video = {
                        'video_id': video_id,
                        'title': video_details.get('title', 'Unknown'),
                        'upload_date': video_details.get('upload_date', ''),
                        'processed': False,
                        'last_check': current_time
                    }
                    new_videos.append(new_video)
                    
                    # Add to DataFrame
                    new_df = pd.DataFrame([new_video])
                    self.df_videos = pd.concat([self.df_videos, new_df], ignore_index=True)
            else:
                # Update last_check for existing video
                self.df_videos.loc[self.df_videos['video_id'] == video_id, 'last_check'] = current_time
        
        # Save updated CSV
        self.df_videos.to_csv(self.video_list_file, index=False)
        
        if new_videos:
            logger.info(f"Found {len(new_videos)} new videos")
            for video in new_videos:
                logger.info(f"  - {video['title']} ({video['video_id']})")
        else:
            logger.info("No new videos found")
        
        return new_videos
    
    def process_unprocessed_videos(self):
        """Process videos that haven't been processed yet"""
        unprocessed = self.df_videos[self.df_videos['processed'] == False]
        
        if unprocessed.empty:
            logger.info("No unprocessed videos found")
            return
        
        logger.info(f"Processing {len(unprocessed)} unprocessed videos")
        
        # Run the batch processing script
        cmd = ['python', 'process_batch.py', '--csv', self.video_list_file, '--full']
        
        logger.info(f"Executing: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
            
            if result.returncode == 0:
                logger.info("Processing completed successfully")
                
                # Reload the CSV to get updated processed status
                self.load_video_list()
            else:
                logger.error(f"Processing failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            logger.error("Processing timed out after 1 hour")
        except Exception as e:
            logger.error(f"Error during processing: {e}")
    
    def cleanup_audio_files(self):
        """Clean up temporary audio files"""
        temp_audio_dir = "temp_audio"
        
        if not os.path.exists(temp_audio_dir):
            return
        
        audio_files = os.listdir(temp_audio_dir)
        cleaned_count = 0
        
        for file in audio_files:
            if file.endswith(('.mp3', '.m4a', '.wav', '.ogg')):
                file_path = os.path.join(temp_audio_dir, file)
                try:
                    os.remove(file_path)
                    cleaned_count += 1
                except Exception as e:
                    logger.warning(f"Could not remove {file}: {e}")
        
        logger.info(f"Cleaned up {cleaned_count} audio files")


def main():
    parser = argparse.ArgumentParser(description='Monitor YouTube channel for new sermons')
    parser.add_argument('--channel', required=True, help='YouTube channel URL or ID')
    parser.add_argument('--process', action='store_true', help='Process new videos')
    parser.add_argument('--cleanup', action='store_true', help='Clean up audio files')
    parser.add_argument('--cookies', type=str, help='Path to cookies file for authentication')
    parser.add_argument('--max-videos', type=int, default=50, help='Maximum number of recent videos to check')
    
    args = parser.parse_args()
    
    # Initialize monitor
    monitor = SermonMonitor(args.channel, args.cookies)
    
    # Check for new videos
    new_videos = monitor.check_for_new_videos()
    
    # Process unprocessed videos if requested
    if args.process:
        monitor.process_unprocessed_videos()
    
    # Clean up audio files if requested
    if args.cleanup:
        monitor.cleanup_audio_files()
    
    logger.info("")  # Add blank line for readability in logs


if __name__ == "__main__":
    main()