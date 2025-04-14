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

def save_transcript(video_id, transcript_data, transcript_dir):
    """
    Save the transcript to a JSON file with timestamps if available.
    
    Args:
        video_id: YouTube video ID
        transcript_data: Transcript data (text and segments)
        transcript_dir: Directory to save the transcript
    
    Returns:
        Path to the saved transcript file
    """
    os.makedirs(transcript_dir, exist_ok=True)
    
    transcript_file = os.path.join(transcript_dir, f"{video_id}.json")
    
    # Check if transcript_data is just text or has structure
    if isinstance(transcript_data, str):
        # Create a structured transcript object from plain text
        transcript_data = {
            "text": transcript_data,
            "segments": []
        }
    
    # Create a structured transcript object
    transcript_obj = {
        "video_id": video_id,
        "text": transcript_data.get("text", ""),
        "segments": transcript_data.get("segments", []),
        "processed_at": datetime.now().isoformat(),
    }
    
    with open(transcript_file, 'w', encoding='utf-8') as f:
        json.dump(transcript_obj, f, ensure_ascii=False, indent=2)
    
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
    Merge multiple transcript chunks with timestamps into a single transcript.
    
    Args:
        transcript_chunks: List of transcript data objects in order
        
    Returns:
        Merged transcript data with adjusted timestamps
    """
    full_text = []
    all_segments = []
    time_offset = 0
    
    for chunk in transcript_chunks:
        if not chunk:
            continue
            
        # Add the text
        full_text.append(chunk.get("text", ""))
        
        # Process segments with adjusted timestamps
        segments = chunk.get("segments", [])
        for segment in segments:
            # Adjust timestamps by the current offset
            if "start" in segment:
                segment["start"] += time_offset
            if "end" in segment:
                segment["end"] += time_offset
            
            all_segments.append(segment)
        
        # Update time offset for next chunk (if available)
        if segments and "end" in segments[-1]:
            time_offset = segments[-1]["end"]
    
    return {
        "text": " ".join(full_text),
        "segments": all_segments
    }

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

def generate_srt_file(video_id, transcript_data, output_dir):
    """
    Generate an SRT subtitle file from transcript data with timestamps.
    
    Args:
        video_id: YouTube video ID
        transcript_data: Transcript data with text and segments
        output_dir: Directory to save the subtitle file
    
    Returns:
        Path to the generated SRT file
    """
    os.makedirs(output_dir, exist_ok=True)
    srt_file = os.path.join(output_dir, f"{video_id}.srt")
    
    # Ensure transcript_data has segments
    segments = transcript_data.get("segments", [])
    if not segments:
        logger.warning(f"No timestamp segments found for {video_id}, cannot generate SRT")
        return None
    
    with open(srt_file, 'w', encoding='utf-8') as f:
        for i, segment in enumerate(segments):
            # SRT index (starting at 1)
            f.write(f"{i+1}\n")
            
            # Format timestamps as HH:MM:SS,mmm --> HH:MM:SS,mmm
            start_time = format_timestamp_srt(segment.get("start", 0))
            end_time = format_timestamp_srt(segment.get("end", 0))
            f.write(f"{start_time} --> {end_time}\n")
            
            # Write segment text
            f.write(f"{segment.get('text', '').strip()}\n\n")
    
    logger.info(f"Generated SRT file for {video_id}: {srt_file}")
    return srt_file

def generate_vtt_file(video_id, transcript_data, output_dir):
    """
    Generate a VTT subtitle file from transcript data with timestamps.
    
    Args:
        video_id: YouTube video ID
        transcript_data: Transcript data with text and segments
        output_dir: Directory to save the subtitle file
    
    Returns:
        Path to the generated VTT file
    """
    os.makedirs(output_dir, exist_ok=True)
    vtt_file = os.path.join(output_dir, f"{video_id}.vtt")
    
    # Ensure transcript_data has segments
    segments = transcript_data.get("segments", [])
    if not segments:
        logger.warning(f"No timestamp segments found for {video_id}, cannot generate VTT")
        return None
    
    with open(vtt_file, 'w', encoding='utf-8') as f:
        # VTT header
        f.write("WEBVTT\n\n")
        
        for segment in segments:
            # Format timestamps as HH:MM:SS.mmm --> HH:MM:SS.mmm
            start_time = format_timestamp_vtt(segment.get("start", 0))
            end_time = format_timestamp_vtt(segment.get("end", 0))
            f.write(f"{start_time} --> {end_time}\n")
            
            # Write segment text
            f.write(f"{segment.get('text', '').strip()}\n\n")
    
    logger.info(f"Generated VTT file for {video_id}: {vtt_file}")
    return vtt_file

def format_timestamp_srt(seconds):
    """
    Format seconds as SRT timestamp: HH:MM:SS,mmm
    """
    hours = int(seconds / 3600)
    minutes = int((seconds % 3600) / 60)
    seconds = seconds % 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    
    return f"{hours:02d}:{minutes:02d}:{int(seconds):02d},{milliseconds:03d}"

def format_timestamp_vtt(seconds):
    """
    Format seconds as VTT timestamp: HH:MM:SS.mmm
    """
    hours = int(seconds / 3600)
    minutes = int((seconds % 3600) / 60)
    seconds = seconds % 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    
    return f"{hours:02d}:{minutes:02d}:{int(seconds):02d}.{milliseconds:03d}"