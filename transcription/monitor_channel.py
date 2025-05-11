#!/usr/bin/env python3
"""
Improved YouTube Channel Monitor
--------------------------------
This script monitors a YouTube channel for new videos, downloads them,
and initiates the transcription and embedding process.

Key improvements:
1. More robust error handling
2. Better logging
3. Improved YouTube API interaction
4. Better comparison of existing vs. new videos
5. Automated retry mechanism for failed downloads
"""

import os
import sys
import csv
import json
import logging
import argparse
import subprocess
import datetime
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("sermon_monitor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("sermon_monitor")

class YouTubeChannelMonitor:
    """Monitor a YouTube channel for new videos and process them"""
    
    def __init__(self, channel_url: str, output_dir: str, csv_file: str, cookies_file: Optional[str] = None):
        """
        Initialize the monitor
        
        Args:
            channel_url: URL of the YouTube channel
            output_dir: Directory to save transcripts and metadata
            csv_file: Path to the CSV file tracking videos
            cookies_file: Optional path to cookies file for authenticated requests
        """
        self.channel_url = channel_url
        self.output_dir = Path(output_dir)
        self.csv_file = Path(csv_file)
        self.cookies_file = cookies_file
        
        # Create directories if they don't exist
        self.transcript_dir = self.output_dir / "transcripts"
        self.metadata_dir = self.output_dir / "metadata"
        
        self.transcript_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing videos
        self.existing_videos = self._load_existing_videos()
        
    def _load_existing_videos(self) -> Dict[str, Dict]:
        """Load existing videos from the CSV file"""
        videos = {}
        
        if not self.csv_file.exists():
            logger.warning(f"CSV file {self.csv_file} does not exist, creating new one")
            with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "video_id", "title", "description", "publish_date", 
                    "duration", "view_count", "like_count", "url", "thumbnail",
                    "processing_status", "processing_date", "transcript_path", "embeddings_status"
                ])
            return {}
        
        try:
            with open(self.csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if 'video_id' in row and row['video_id']:
                        videos[row['video_id']] = row
                        
            logger.info(f"Loaded {len(videos)} existing videos from {self.csv_file}")
            return videos
            
        except Exception as e:
            logger.error(f"Error loading CSV file: {e}")
            return {}
    
    def _save_videos_to_csv(self, videos: Dict[str, Dict]):
        """Save videos dictionary to CSV file"""
        try:
            fieldnames = [
                "video_id", "title", "description", "publish_date", 
                "duration", "view_count", "like_count", "url", "thumbnail",
                "processing_status", "processing_date", "transcript_path", "embeddings_status"
            ]
            
            # Make a backup of the existing file
            if self.csv_file.exists():
                backup_file = self.csv_file.with_suffix('.csv.bak')
                import shutil
                shutil.copy(self.csv_file, backup_file)
            
            with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for video_data in videos.values():
                    writer.writerow(video_data)
                    
            logger.info(f"Saved {len(videos)} videos to {self.csv_file}")
            
        except Exception as e:
            logger.error(f"Error saving to CSV file: {e}")
            raise
    
    def _run_yt_dlp(self, args: List[str]) -> Tuple[bool, str]:
        """Run yt-dlp with given arguments and return output"""
        try:
            # Build the full command
            command = ["yt-dlp"] + args
            
            # Add cookies if provided
            if self.cookies_file:
                command.extend(["--cookies", self.cookies_file])
            
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
            
    def fetch_channel_videos(self) -> List[Dict]:
        """Fetch videos from the channel using yt-dlp"""
        # Arguments for yt-dlp to get channel metadata in JSON format
        args = [
            "--dump-json",  # Output video metadata as JSON
            "--flat-playlist",  # Don't download videos
            "--playlist-end", "20",  # Limit to last 20 videos (adjust as needed)
            "--no-warnings",  # Reduce noise in output
            self.channel_url  # Channel URL
        ]
        
        # Try up to 3 times
        for attempt in range(3):
            try:
                success, output = self._run_yt_dlp(args)
                
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
    
    def get_video_details(self, video_id: str) -> Optional[Dict]:
        """Get detailed metadata for a specific video"""
        args = [
            "--dump-json",
            f"https://www.youtube.com/watch?v={video_id}"
        ]
        
        success, output = self._run_yt_dlp(args)
        
        if not success:
            logger.error(f"Failed to get details for video {video_id}")
            return None
        
        try:
            video_info = json.loads(output)
            return video_info
        except json.JSONDecodeError:
            logger.error(f"Could not parse JSON for video {video_id}")
            return None
    
    def download_and_transcribe(self, video_id: str) -> bool:
        """Download audio and transcribe the video"""
        logger.info(f"Processing video {video_id}")
        
        # First, update the status in the CSV to show we're working on it
        if video_id in self.existing_videos:
            self.existing_videos[video_id]["processing_status"] = "in_progress"
            self.existing_videos[video_id]["processing_date"] = datetime.datetime.now().isoformat()
            self._save_videos_to_csv(self.existing_videos)
        
        # Build the file paths
        audio_file = f"temp_{video_id}.m4a"
        transcript_file = self.transcript_dir / f"{video_id}.json"
        
        try:
            # Step 1: Download audio
            logger.info(f"Downloading audio for {video_id}")
            audio_args = [
                "--extract-audio",
                "--audio-format", "m4a",
                "--audio-quality", "0",  # Best quality
                "-o", audio_file,
                f"https://www.youtube.com/watch?v={video_id}"
            ]
            
            success, _ = self._run_yt_dlp(audio_args)
            if not success:
                logger.error(f"Failed to download audio for {video_id}")
                self._update_video_status(video_id, "failed")
                return False
            
            # Step 2: Transcribe using OpenAI Whisper
            logger.info(f"Transcribing audio for {video_id}")
            
            # Check if an OpenAI API key is available
            if not os.environ.get("OPENAI_API_KEY"):
                logger.error("No OpenAI API key found in environment")
                self._update_video_status(video_id, "failed")
                return False
                
            # Use the whisper-1 model via OpenAI API
            import openai
            
            client = openai.OpenAI()
            
            with open(audio_file, "rb") as audio:
                transcription = client.audio.transcriptions.create(
                    file=audio,
                    model="whisper-1",
                    response_format="verbose_json"
                )
            
            # Save the transcription
            with open(transcript_file, 'w', encoding='utf-8') as f:
                json.dump(transcription, f, indent=2)
            
            # Update CSV with success
            self._update_video_status(
                video_id, 
                "processed", 
                transcript_path=str(transcript_file)
            )
            
            logger.info(f"Successfully transcribed video {video_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing video {video_id}: {e}")
            self._update_video_status(video_id, "failed")
            return False
            
        finally:
            # Clean up temporary audio file
            if os.path.exists(audio_file):
                try:
                    os.remove(audio_file)
                except:
                    pass
    
    def _update_video_status(self, video_id: str, status: str, transcript_path: Optional[str] = None):
        """Update the processing status of a video in the CSV"""
        if video_id in self.existing_videos:
            self.existing_videos[video_id]["processing_status"] = status
            self.existing_videos[video_id]["processing_date"] = datetime.datetime.now().isoformat()
            
            if transcript_path:
                self.existing_videos[video_id]["transcript_path"] = transcript_path
                
            # If processed successfully, mark for embedding
            if status == "processed":
                self.existing_videos[video_id]["embeddings_status"] = "pending"
                
            self._save_videos_to_csv(self.existing_videos)
    
    def process_new_videos(self):
        """Main function to check for and process new videos"""
        logger.info("Starting to check for new videos")
        
        # Fetch channel videos
        channel_videos = self.fetch_channel_videos()
        
        if not channel_videos:
            logger.warning("No videos found in channel or error fetching videos")
            return
        
        # Check for new videos
        new_videos = []
        updated_videos = []
        
        for video in channel_videos:
            video_id = video.get('id')
            if not video_id:
                continue
                
            if video_id not in self.existing_videos:
                # This is a new video
                logger.info(f"Found new video: {video_id} - {video.get('title', 'Unknown Title')}")
                
                # Get detailed info
                detailed_info = self.get_video_details(video_id)
                
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
                
                self.existing_videos[video_id] = video_data
                new_videos.append(video_id)
                
            elif self.existing_videos[video_id]["processing_status"] in ["failed", "pending"]:
                # This is a video that needs to be reprocessed
                logger.info(f"Video needs processing: {video_id} - {self.existing_videos[video_id].get('title', 'Unknown Title')}")
                updated_videos.append(video_id)
        
        # Save updated CSV
        self._save_videos_to_csv(self.existing_videos)
        
        logger.info(f"Found {len(new_videos)} new videos and {len(updated_videos)} videos to reprocess")
        
        # Process new videos
        for video_id in new_videos + updated_videos:
            success = self.download_and_transcribe(video_id)
            if not success:
                logger.warning(f"Failed to process video {video_id}")
            else:
                logger.info(f"Successfully processed video {video_id}")
            
            # Wait a bit between processing videos to avoid rate limits
            time.sleep(5)
        
        logger.info("Finished processing videos")

def generate_embeddings(transcript_dir: str, csv_file: str):
    """Generate embeddings for transcripts"""
    logger.info("Starting embedding generation")
    
    try:
        # Call the embedding script
        subprocess.run(
            ["python", "tools/transcript_to_embeddings.py", "--skip_existing"],
            check=True
        )
        logger.info("Embedding generation completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error generating embeddings: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during embedding: {e}")
        return False

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Monitor YouTube channel for new sermon videos")
    parser.add_argument("--channel", required=True, help="YouTube channel URL")
    parser.add_argument("--output-dir", default="transcription/data", help="Output directory for transcripts and metadata")
    parser.add_argument("--csv-file", default="transcription/data/video_list.csv", help="CSV file to track videos")
    parser.add_argument("--cookies", help="Path to YouTube cookies file")
    parser.add_argument("--process", action="store_true", help="Process new videos (download and transcribe)")
    parser.add_argument("--embed", action="store_true", help="Generate embeddings after processing")
    
    args = parser.parse_args()
    
    # Create the monitor
    monitor = YouTubeChannelMonitor(
        channel_url=args.channel,
        output_dir=args.output_dir,
        csv_file=args.csv_file,
        cookies_file=args.cookies
    )
    
    if args.process:
        # Check for new videos and process them
        monitor.process_new_videos()
    
    if args.embed:
        # Generate embeddings for processed videos
        generate_embeddings(
            transcript_dir=os.path.join(args.output_dir, "transcripts"),
            csv_file=args.csv_file
        )

if __name__ == "__main__":
    main()