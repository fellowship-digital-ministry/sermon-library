"""
Utility functions for the sermon transcription pipeline.
"""
import os
import csv
import json
import logging
import re
import math
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("sermon_processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def extract_video_id(youtube_url):
    """
    Extract the video ID from a YouTube URL.
    
    Args:
        youtube_url: YouTube URL in various possible formats
        
    Returns:
        YouTube video ID or None if no valid ID found
    """
    # Patterns for YouTube URLs
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/)([^&\n?#]+)',  # Standard and shortened URLs
        r'youtube\.com\/embed\/([^&\n?#]+)',                     # Embedded URLs
        r'youtube\.com\/v\/([^&\n?#]+)',                         # Old embedded URLs
    ]
    
    for pattern in patterns:
        match = re.search(pattern, youtube_url)
        if match:
            return match.group(1)
    
    logger.warning(f"Could not extract video ID from URL: {youtube_url}")
    return None

def load_video_list(file_path):
    """
    Load the list of YouTube videos from a CSV file.
    Expected format: video_id,title,publish_date,description
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        List of dictionaries with video information
    """
    if not os.path.exists(file_path):
        # Create a sample file if it doesn't exist
        create_sample_video_list(file_path)
        logger.info(f"Created sample video list at {file_path}")
    
    videos = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            videos.append(row)
    
    logger.info(f"Loaded {len(videos)} videos from {file_path}")
    return videos

def create_sample_video_list(file_path):
    """
    Create a sample video list CSV file for demonstration.
    
    Args:
        file_path: Path to save the CSV file
    """
    # Example sermon videos from a church YouTube channel
    sample_videos = [
        {
            "video_id": "jvCTaccEkMk", 
            "title": "Example Sermon 1",
            "publish_date": "2023-01-01",
            "description": "A sample sermon description"
        },
        {
            "video_id": "eX8K1L3dNMU", 
            "title": "Example Sermon 2",
            "publish_date": "2023-01-08",
            "description": "Another sample sermon description"
        },
        {
            "video_id": "5THRj9qoiy8", 
            "title": "Example Sermon 3",
            "publish_date": "2023-01-15",
            "description": "Yet another sample sermon description"
        },
        {
            "video_id": "x2pKqB8Kx8A", 
            "title": "Example Sermon 4",
            "publish_date": "2023-01-22",
            "description": "Fourth sample sermon description"
        },
        {
            "video_id": "Pdmic1hFOME", 
            "title": "Example Sermon 5",
            "publish_date": "2023-01-29",
            "description": "Fifth sample sermon description"
        }
    ]
    
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ["video_id", "title", "publish_date", "description"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for video in sample_videos:
            writer.writerow(video)

def save_transcript(video_id, transcript, transcript_dir):
    """
    Save the transcript to a JSON file.
    
    Args:
        video_id: YouTube video ID
        transcript: Transcript text
        transcript_dir: Directory to save the transcript
    
    Returns:
        Path to the saved transcript file
    """
    os.makedirs(transcript_dir, exist_ok=True)
    
    transcript_file = os.path.join(transcript_dir, f"{video_id}.json")
    
    # Create a structured transcript object
    transcript_data = {
        "video_id": video_id,
        "text": transcript,
        "processed_at": datetime.now().isoformat(),
    }
    
    with open(transcript_file, 'w', encoding='utf-8') as f:
        json.dump(transcript_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved transcript for {video_id} to {transcript_file}")
    return transcript_file

def save_metadata(video_id, metadata, metadata_dir):
    """
    Save video metadata to a JSON file.
    
    Args:
        video_id: YouTube video ID
        metadata: Video metadata
        metadata_dir: Directory to save metadata
    
    Returns:
        Path to the saved metadata file
    """
    os.makedirs(metadata_dir, exist_ok=True)
    
    metadata_file = os.path.join(metadata_dir, f"{video_id}_metadata.json")
    
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved metadata for {video_id} to {metadata_file}")
    return metadata_file

def get_video_url(video_id):
    """
    Get the YouTube URL for a video ID.
    
    Args:
        video_id: YouTube video ID
    
    Returns:
        YouTube URL
    """
    return f"https://www.youtube.com/watch?v={video_id}"

def get_audio_filename(video_id, audio_dir):
    """
    Get the audio filename for a video ID.
    
    Args:
        video_id: YouTube video ID
        audio_dir: Directory for audio files
    
    Returns:
        Audio file path
    """
    return os.path.join(audio_dir, f"{video_id}.mp3")

def split_audio_file(audio_file, chunk_duration_seconds=600, output_dir=None):
    """
    Split a large audio file into smaller chunks to stay under the Whisper API limit.
    Uses pydub to split the file without re-encoding.
    
    Args:
        audio_file: Path to the audio file
        chunk_duration_seconds: Duration of each chunk in seconds (default: 10 minutes)
        output_dir: Directory to save chunks (default: same as input file)
    
    Returns:
        List of paths to the chunk files
    """
    import os
    from pydub import AudioSegment
    
    if output_dir is None:
        output_dir = os.path.dirname(audio_file)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the base filename without extension
    base_name = os.path.splitext(os.path.basename(audio_file))[0]
    
    try:
        # Load audio file
        logger.info(f"Loading audio file for splitting: {audio_file}")
        audio = AudioSegment.from_file(audio_file)
        
        # Calculate number of chunks
        total_duration_seconds = len(audio) / 1000
        num_chunks = math.ceil(total_duration_seconds / chunk_duration_seconds)
        
        logger.info(f"Splitting {audio_file} into {num_chunks} chunks of {chunk_duration_seconds} seconds")
        
        chunk_files = []
        
        # Split the file into chunks
        for i in range(num_chunks):
            start_ms = i * chunk_duration_seconds * 1000
            end_ms = min((i + 1) * chunk_duration_seconds * 1000, len(audio))
            
            chunk = audio[start_ms:end_ms]
            chunk_file = os.path.join(output_dir, f"{base_name}_chunk_{i+1}.mp3")
            
            chunk.export(chunk_file, format="mp3")
            chunk_files.append(chunk_file)
            
            logger.info(f"Created chunk {i+1}/{num_chunks}: {chunk_file}")
        
        return chunk_files
    
    except Exception as e:
        logger.error(f"Failed to split audio file {audio_file}: {str(e)}")
        return []

def merge_transcripts(transcript_chunks):
    """
    Merge multiple transcript chunks into a single transcript.
    
    Args:
        transcript_chunks: List of transcript texts in order
        
    Returns:
        Merged transcript text
    """
    return " ".join(transcript_chunks)

def get_file_size_mb(file_path):
    """
    Get the size of a file in megabytes.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File size in MB
    """
    import os
    return os.path.getsize(file_path) / (1024 * 1024)