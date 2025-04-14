"""
Script to transcribe audio files using OpenAI's Whisper API.
"""
import os
import time
import logging
from openai import OpenAI
from tqdm import tqdm

import config
from utils import save_transcript, save_metadata, split_audio_file, merge_transcripts, get_file_size_mb

logger = logging.getLogger(__name__)

# Maximum file size for Whisper API in MB (with some buffer)
MAX_FILE_SIZE_MB = 24

def transcribe_audio(audio_file, model=config.WHISPER_MODEL, language=config.WHISPER_LANGUAGE):
    """
    Transcribe an audio file using OpenAI's Whisper API.
    Handles large files by splitting them into chunks.
    
    Args:
        audio_file: Path to the audio file
        model: Whisper model to use
        language: Language code
    
    Returns:
        Transcript text or None if transcription failed
    """
    if not os.path.exists(audio_file):
        logger.error(f"Audio file {audio_file} does not exist")
        return None
    
    # Check file size
    file_size_mb = get_file_size_mb(audio_file)
    logger.info(f"Audio file size: {file_size_mb:.2f} MB")
    
    # If file is too large, split it
    if file_size_mb > MAX_FILE_SIZE_MB:
        logger.info(f"File exceeds maximum size ({MAX_FILE_SIZE_MB} MB), splitting into chunks")
        chunk_files = split_audio_file(audio_file)
        
        if not chunk_files:
            logger.error("Failed to split audio file")
            return None
        
        # Process each chunk
        transcripts = []
        for i, chunk_file in enumerate(tqdm(chunk_files, desc="Transcribing chunks")):
            logger.info(f"Transcribing chunk {i+1}/{len(chunk_files)}")
            chunk_transcript = transcribe_chunk(chunk_file, model, language)
            
            if chunk_transcript:
                transcripts.append(chunk_transcript)
            else:
                logger.error(f"Failed to transcribe chunk {i+1}")
        
        # Merge transcripts
        if transcripts:
            full_transcript = merge_transcripts(transcripts)
            logger.info(f"Successfully transcribed all chunks ({len(full_transcript)} characters)")
            return full_transcript
        else:
            logger.error("No chunks were successfully transcribed")
            return None
    else:
        # File is small enough, transcribe directly
        return transcribe_chunk(audio_file, model, language)

def transcribe_chunk(audio_file, model, language):
    """
    Transcribe a single audio chunk using the Whisper API.
    
    Args:
        audio_file: Path to the audio file
        model: Whisper model to use
        language: Language code
    
    Returns:
        Transcript text or None if transcription failed
    """
    logger.info(f"Transcribing {audio_file} with model {model}")
    
    client = OpenAI(api_key=config.OPENAI_API_KEY)
    
    try:
        with open(audio_file, "rb") as audio:
            # Call the OpenAI API
            response = client.audio.transcriptions.create(
                model=model,
                file=audio,
                language=language
            )
        
        # Extract transcript
        transcript = response.text
        
        logger.info(f"Successfully transcribed {audio_file} ({len(transcript)} characters)")
        return transcript
    except Exception as e:
        logger.error(f"Failed to transcribe {audio_file}: {str(e)}")
        return None

def transcribe_batch(audio_files, output_dir=config.TRANSCRIPT_DIR, metadata_dir=config.METADATA_DIR):
    """
    Transcribe a batch of audio files.
    
    Args:
        audio_files: Dictionary mapping video IDs to audio file paths
        output_dir: Directory to save transcripts
        metadata_dir: Directory to save metadata
    
    Returns:
        Dictionary mapping video IDs to transcript file paths
    """
    results = {}
    
    for video_id, audio_file in tqdm(audio_files.items(), desc="Transcribing audio"):
        # Add a small delay to avoid API rate limits
        time.sleep(1)
        
        transcript = transcribe_audio(audio_file)
        
        if transcript:
            # Save transcript
            transcript_file = save_transcript(video_id, transcript, output_dir)
            results[video_id] = transcript_file
            
            # Save metadata about the transcription process
            metadata = {
                "video_id": video_id,
                "audio_file": audio_file,
                "transcript_file": transcript_file,
                "transcription_timestamp": time.time(),
                "model": config.WHISPER_MODEL,
                "language": config.WHISPER_LANGUAGE,
                "transcript_length": len(transcript)
            }
            save_metadata(video_id, metadata, metadata_dir)
    
    logger.info(f"Transcribed {len(results)} audio files")
    return results

if __name__ == "__main__":
    # This allows running this script directly for testing
    import sys
    
    if len(sys.argv) > 1:
        # Transcribe a single file
        audio_file = sys.argv[1]
        transcript = transcribe_audio(audio_file)
        if transcript:
            print(transcript[:500] + "...")  # Print beginning of transcript
    else:
        print("Usage: python transcribe_audio.py <audio_file>")