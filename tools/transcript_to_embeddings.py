"""
Generate embeddings from sermon transcripts and store them in Pinecone.
CSV-driven approach: Uses video_list.csv as the source of truth.

Improvements:
1. Better error handling
2. More comprehensive logging
3. Automatic retry functionality
4. More robust CSV processing
5. Additional monitoring capabilities
"""

import os
import json
import time
import argparse
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
import logging
from datetime import datetime
from tqdm import tqdm
import sys
import traceback

import openai
from pinecone import Pinecone, ServerlessSpec, CloudProvider

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("embedding_processing.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('sermon_embeddings')

# Constants - use environment variables directly
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT", "us-east-1")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "sermon-embeddings")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "250"))  # Number of words per chunk
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "50"))  # Number of overlapping words between chunks
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "10"))  # Number of embeddings to send at once
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "3"))  # Maximum number of retry attempts

def check_environment():
    """Verify required environment variables are set."""
    missing_vars = []
    for var in ["OPENAI_API_KEY", "PINECONE_API_KEY"]:
        if not os.environ.get(var):
            missing_vars.append(var)
    
    if missing_vars:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

def initialize_clients():
    """Initialize the OpenAI and Pinecone clients."""
    # Initialize OpenAI client
    openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
    
    # Initialize Pinecone with new API
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Check if index exists
    index_exists = False
    try:
        for index in pc.list_indexes():
            if index.name == PINECONE_INDEX_NAME:
                index_exists = True
                break
    except Exception as e:
        logger.error(f"Error checking Pinecone indexes: {e}")
        raise
    
    # Create index if it doesn't exist
    if not index_exists:
        logger.info(f"Creating new Pinecone index: {PINECONE_INDEX_NAME}")
        try:
            # Create a new index
            index_config = pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=1536,  # text-embedding-3-small uses 1536 dimensions
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=CloudProvider.AWS,
                    region=PINECONE_ENVIRONMENT
                )
            )
            logger.info(f"Created new index with host: {index_config.host}")
        except Exception as e:
            logger.error(f"Error creating Pinecone index: {e}")
            raise
    
    # Connect to the index
    try:
        pinecone_index = pc.Index(PINECONE_INDEX_NAME)
        # Test the connection
        stats = pinecone_index.describe_index_stats()
        logger.info(f"Connected to Pinecone index. Vector count: {stats.get('total_vector_count', 'unknown')}")
        
        return openai_client, pinecone_index
    except Exception as e:
        logger.error(f"Error connecting to Pinecone index: {e}")
        raise

def load_video_list_csv(csv_path: str) -> pd.DataFrame:
    """
    Load the video_list.csv to determine which videos need embedding.
    This is the source of truth for the system.
    """
    if not os.path.exists(csv_path):
        logger.error(f"Video list CSV not found at {csv_path}")
        raise FileNotFoundError(f"Video list CSV not found at {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded video list CSV with {len(df)} rows")
        
        # Add embeddings_status column if it doesn't exist
        if 'embeddings_status' not in df.columns:
            df['embeddings_status'] = 'pending'
            logger.info("Added 'embeddings_status' column to video list")
        
        # Add embeddings_date column if it doesn't exist
        if 'embeddings_date' not in df.columns:
            df['embeddings_date'] = None
            logger.info("Added 'embeddings_date' column to video list")
            
        # Add embeddings_count column if it doesn't exist
        if 'embeddings_count' not in df.columns:
            df['embeddings_count'] = 0
            logger.info("Added 'embeddings_count' column to video list")
            
        # Ensure values are of the right types
        df['embeddings_count'] = pd.to_numeric(df['embeddings_count'], errors='coerce').fillna(0).astype(int)
        
        return df
    except Exception as e:
        logger.error(f"Error loading video list CSV: {e}")
        raise

def save_video_list_csv(df: pd.DataFrame, csv_path: str):
    """Save updated video list CSV"""
    try:
        # Create a backup
        backup_path = f"{csv_path}.bak"
        if os.path.exists(csv_path):
            import shutil
            shutil.copy(csv_path, backup_path)
            logger.info(f"Created backup of CSV at {backup_path}")
        
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved updated video list CSV with {len(df)} rows")
    except Exception as e:
        logger.error(f"Error saving video list CSV: {e}")
        raise

def get_videos_needing_embeddings(df_videos: pd.DataFrame) -> List[Dict]:
    """
    Get list of videos that need embeddings based on CSV status
    Returns list of video metadata dictionaries with all the information needed
    """
    # Get videos that are processed but don't have embeddings yet
    videos_to_process = df_videos[
        (df_videos['processing_status'] == 'processed') & 
        ((df_videos['embeddings_status'] != 'completed') | 
         (df_videos['embeddings_status'].isna()))
    ]
    
    if videos_to_process.empty:
        logger.info("No videos need embeddings processing")
        return []
    
    logger.info(f"Found {len(videos_to_process)} videos needing embeddings")
    
    # Convert to list of dictionaries
    videos_list = []
    for _, row in videos_to_process.iterrows():
        video_data = {
            'video_id': row['video_id'],
            'title': row['title'],
            'transcript_path': row.get('transcript_path', ''),
            'publish_date': row.get('publish_date', '')
        }
        videos_list.append(video_data)
    
    return videos_list

def load_transcript(file_path: str) -> Dict:
    """Load and parse a transcript JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Basic validation
        if not isinstance(data, dict):
            raise ValueError(f"Transcript file does not contain a JSON object: {file_path}")
            
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing transcript JSON at {file_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading transcript from {file_path}: {e}")
        raise

def get_transcript_path(video_id: str, transcript_dir: str, csv_transcript_path: str = None) -> str:
    """Get the path to a transcript file, checking multiple possible locations"""
    # First try the path from CSV if provided
    if csv_transcript_path and os.path.exists(csv_transcript_path):
        return csv_transcript_path
    
    # Try default location
    default_path = os.path.join(transcript_dir, f"{video_id}.json")
    if os.path.exists(default_path):
        return default_path
    
    # Try alternative locations
    alt_path = os.path.join("transcription", "data", "transcripts", f"{video_id}.json")
    if os.path.exists(alt_path):
        return alt_path
    
    # Not found
    return None

def extract_video_metadata(transcript: Dict, video_data: Dict) -> Dict:
    """Extract metadata from the transcript and video data from CSV"""
    video_id = transcript.get('video_id', video_data.get('video_id', ''))
    
    # If we have metadata files, we could load additional information here
    metadata_path = os.path.join('transcription', 'data', 'metadata', f"{video_id}.json")
    metadata = {}
    
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load metadata file for {video_id}: {e}")
    
    # Use metadata from both sources, prioritizing CSV data
    return {
        "video_id": video_id,
        "title": video_data.get('title') or metadata.get('title', f"Sermon {video_id}"),
        "channel": metadata.get('channel', 'Unknown'),
        "publish_date": video_data.get('publish_date') or metadata.get('upload_date', datetime.now().strftime('%Y-%m-%d')),
        "url": metadata.get('webpage_url') or f"https://www.youtube.com/watch?v={video_id}"
    }

def chunk_transcript(transcript: Dict) -> List[Dict]:
    """
    Split the transcript into chunks of appropriate size for embedding.
    Uses both text and time information to create meaningful chunks.
    """
    full_text = transcript.get('text', '')
    segments = transcript.get('segments', [])
    
    # If no segments, use a simple text chunking approach
    if not segments:
        words = full_text.split()
        chunks = []
        
        for i in range(0, len(words), CHUNK_SIZE - CHUNK_OVERLAP):
            chunk_words = words[i:i + CHUNK_SIZE]
            chunk_text = ' '.join(chunk_words)
            chunks.append({
                "text": chunk_text,
                "start_time": 0,
                "end_time": 0,
                "segment_ids": []
            })
        
        return chunks
    
    # If we have segments, create chunks that respect segment boundaries
    chunks = []
    current_chunk = {
        "text": "",
        "start_time": segments[0]["start"] if segments else 0,
        "end_time": segments[0]["end"] if segments else 0,
        "segment_ids": []
    }
    current_word_count = 0
    
    for segment in segments:
        segment_text = segment["text"].strip()
        segment_words = segment_text.split()
        segment_word_count = len(segment_words)
        
        # If adding this segment would exceed the chunk size, save the current chunk and start a new one
        if current_word_count + segment_word_count > CHUNK_SIZE:
            # Only save if we have content
            if current_word_count > 0:
                chunks.append(current_chunk)
            
            # Start a new chunk with overlap
            overlap_text = ""
            overlap_ids = []
            
            # Calculate overlap from previous chunk
            if current_word_count > CHUNK_OVERLAP:
                # Get the last CHUNK_OVERLAP words from the previous chunk
                overlap_words = current_chunk["text"].split()[-CHUNK_OVERLAP:]
                overlap_text = " ".join(overlap_words)
                overlap_ids = current_chunk["segment_ids"][-1:]  # Include at least the last segment ID
            
            current_chunk = {
                "text": overlap_text,
                "start_time": segment["start"],
                "end_time": segment["end"],
                "segment_ids": overlap_ids.copy()
            }
            current_word_count = len(overlap_text.split())
        
        # Add the current segment to the chunk
        if current_chunk["text"]:
            current_chunk["text"] += " "
        current_chunk["text"] += segment_text
        current_chunk["end_time"] = segment["end"]
        current_chunk["segment_ids"].append(segment.get("id", len(current_chunk["segment_ids"])))
        current_word_count += segment_word_count
    
    # Add the last chunk if it's not empty
    if current_word_count > 0:
        chunks.append(current_chunk)
    
    return chunks

def create_embeddings(client, text_chunks: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of text chunks using OpenAI's API.
    Returns a list of embedding vectors.
    """
    embeddings = []
    
    # Process in batches to avoid rate limits
    for i in range(0, len(text_chunks), BATCH_SIZE):
        batch = text_chunks[i:i + BATCH_SIZE]
        
        for retry in range(MAX_RETRIES):
            try:
                response = client.embeddings.create(
                    input=batch,
                    model=EMBEDDING_MODEL
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                
                # Avoid rate limiting
                if i + BATCH_SIZE < len(text_chunks):
                    time.sleep(0.5)
                
                # If successful, break the retry loop
                break
                
            except Exception as e:
                logger.error(f"Error generating embeddings for batch {i} (attempt {retry+1}/{MAX_RETRIES}): {str(e)}")
                if retry < MAX_RETRIES - 1:
                    # Wait longer between retries
                    wait_time = (retry + 1) * 2
                    logger.info(f"Waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
                else:
                    # Insert empty embeddings as placeholders on last retry
                    logger.error(f"Failed to generate embeddings after {MAX_RETRIES} attempts")
                    embeddings.extend([[] for _ in range(len(batch))])
    
    return embeddings

def upload_to_pinecone(index, embeddings: List[List[float]], chunks: List[Dict], metadata: Dict) -> int:
    """
    Upload embeddings and metadata to Pinecone.
    Returns the number of successfully uploaded vectors.
    """
    vectors = []
    
    for i, (embedding, chunk) in enumerate(zip(embeddings, chunks)):
        if not embedding:  # Skip failed embeddings
            continue
            
        # Create a unique ID for each chunk
        vector_id = f"{metadata['video_id']}_{i}"
        
        # Convert segment_ids to list of strings if present
        segment_ids = [str(id) for id in chunk["segment_ids"]] if chunk["segment_ids"] else []
        
        # Combine chunk metadata with video metadata
        vector_metadata = {
            **metadata,
            "chunk_index": i,
            "text": chunk["text"],
            "start_time": chunk["start_time"],
            "end_time": chunk["end_time"],
            "segment_ids": segment_ids,
            "word_count": len(chunk["text"].split())
        }
        
        vectors.append({
            "id": vector_id,
            "values": embedding,
            "metadata": vector_metadata
        })
    
    # Log the number of vectors to upload
    logger.info(f"Preparing to upload {len(vectors)} vectors to Pinecone for {metadata['video_id']}")
    
    # Upload vectors in batches
    uploaded = 0
    for i in range(0, len(vectors), BATCH_SIZE):
        batch = vectors[i:i + BATCH_SIZE]
        for retry in range(MAX_RETRIES):
            try:
                logger.info(f"Uploading batch {i//BATCH_SIZE + 1} of {(len(vectors) + BATCH_SIZE - 1)//BATCH_SIZE}")
                index.upsert(vectors=batch)
                uploaded += len(batch)
                logger.info(f"Successfully uploaded {uploaded}/{len(vectors)} vectors")
                
                # Avoid rate limiting
                if i + BATCH_SIZE < len(vectors):
                    time.sleep(0.5)
                
                # If successful, break retry loop
                break
                
            except Exception as e:
                error_message = f"Error uploading batch {i} to Pinecone (attempt {retry+1}/{MAX_RETRIES}): {str(e)}"
                logger.error(error_message)
                if retry < MAX_RETRIES - 1:
                    # Wait longer between retries
                    wait_time = (retry + 1) * 2
                    logger.info(f"Waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to upload batch after {MAX_RETRIES} attempts")
    
    return uploaded

def check_existing_embeddings(pinecone_index, video_id: str) -> bool:
    """Check if embeddings already exist for this video"""
    try:
        # Query for any vectors with this video_id
        response = pinecone_index.query(
            vector=[0.0] * 1536,  # Dummy vector
            filter={"video_id": video_id},
            top_k=1,
            include_metadata=False
        )
        # If we get any matches, embeddings exist
        return len(response.matches) > 0
    except Exception as e:
        logger.warning(f"Error checking existing embeddings for {video_id}: {e}")
        return False

def process_video(
    video_data: Dict, 
    transcript_dir: str, 
    openai_client, 
    pinecone_index,
    skip_existing: bool = True
) -> Tuple[bool, int, int]:
    """
    Process a single video's transcript and generate embeddings.
    Returns success status, number of chunks, and number of vectors uploaded.
    """
    video_id = video_data['video_id']
    logger.info(f"Processing video: {video_id} - {video_data['title']}")
    
    # Check if embeddings already exist
    if skip_existing and check_existing_embeddings(pinecone_index, video_id):
        logger.info(f"Skipping {video_id} - embeddings already exist in Pinecone")
        return True, 0, 0
    
    # Get transcript path
    transcript_path = get_transcript_path(
        video_id, 
        transcript_dir, 
        video_data.get('transcript_path', '')
    )
    
    if not transcript_path:
        logger.warning(f"Transcript not found for video {video_id}")
        return False, 0, 0
    
    try:
        # Load transcript
        transcript = load_transcript(transcript_path)
        
        # Extract metadata
        metadata = extract_video_metadata(transcript, video_data)
        
        # Split transcript into chunks
        chunks = chunk_transcript(transcript)
        
        if not chunks:
            logger.warning(f"No chunks generated for {video_id}")
            return False, 0, 0
        
        # Get just the text from each chunk for embedding
        texts = [chunk["text"] for chunk in chunks]
        
        # Generate embeddings
        embeddings = create_embeddings(openai_client, texts)
        
        # Upload to Pinecone
        uploaded = upload_to_pinecone(pinecone_index, embeddings, chunks, metadata)
        
        logger.info(f"Successfully processed {video_id}: {len(chunks)} chunks, {uploaded} vectors uploaded")
        return True, len(chunks), uploaded
        
    except Exception as e:
        logger.error(f"Error processing {video_id}: {str(e)}")
        traceback.print_exc()
        return False, 0, 0

def process_videos_from_csv(
    csv_path: str,
    transcript_dir: str,
    skip_existing: bool = True,
    limit: Optional[int] = None
) -> Tuple[int, int]:
    """
    Process videos based on their status in the CSV file.
    Returns total chunks and vectors uploaded.
    """
    try:
        # Check environment
        check_environment()
        
        # Initialize clients
        openai_client, pinecone_index = initialize_clients()
        
        # Load video list CSV
        df_videos = load_video_list_csv(csv_path)
        
        # Get videos that need embeddings
        videos_to_process = get_videos_needing_embeddings(df_videos)
        
        if not videos_to_process:
            logger.info("No videos need embeddings processing")
            return 0, 0
        
        # Apply limit if specified
        if limit and limit > 0 and len(videos_to_process) > limit:
            logger.info(f"Limiting processing to {limit} videos")
            videos_to_process = videos_to_process[:limit]
        
        # Track totals
        total_chunks = 0
        total_uploaded = 0
        
        # Process each video
        for video_data in tqdm(videos_to_process, desc="Processing videos"):
            video_id = video_data['video_id']
            
            # Process the video
            success, chunks, uploaded = process_video(
                video_data,
                transcript_dir,
                openai_client,
                pinecone_index,
                skip_existing
            )
            
            # Update the CSV regardless of success
            if video_id in df_videos['video_id'].values:
                idx = df_videos.index[df_videos['video_id'] == video_id].tolist()[0]
                if success and uploaded > 0:
                    df_videos.at[idx, 'embeddings_status'] = 'completed'
                    df_videos.at[idx, 'embeddings_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    df_videos.at[idx, 'embeddings_count'] = uploaded
                elif not success:
                    df_videos.at[idx, 'embeddings_status'] = 'failed'
            
            # Track totals
            total_chunks += chunks
            total_uploaded += uploaded
            
            # Save the CSV after each video to preserve progress
            save_video_list_csv(df_videos, csv_path)
        
        logger.info(f"All videos processed. Generated {total_chunks} chunks and uploaded {total_uploaded} vectors.")
        return total_chunks, total_uploaded
        
    except Exception as e:
        logger.error(f"Error in batch processing: {str(e)}")
        traceback.print_exc()
        return 0, 0

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Generate embeddings from sermon transcripts")
    parser.add_argument("--video_list_csv", type=str, default="./transcription/data/video_list.csv",
                        help="Path to video_list.csv file")
    parser.add_argument("--transcript_dir", type=str, default="./transcription/data/transcripts",
                        help="Directory containing transcript JSON files")
    parser.add_argument("--skip_existing", action="store_true", default=True,
                        help="Skip videos that already have embeddings in Pinecone")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit the number of videos to process")
    parser.add_argument("--api_key", type=str, default=None,
                        help="Pinecone API key (overrides environment variable)")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="Enable verbose logging")
    args = parser.parse_args()
    
    # Set log level based on verbose flag
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Override environment variables with command line arguments if provided
    if args.api_key:
        os.environ["PINECONE_API_KEY"] = args.api_key
    
    # Verify environment
    try:
        check_environment()
    except EnvironmentError as e:
        logger.error(str(e))
        print(f"Error: {str(e)}")
        print("Please set the required environment variables or provide API keys as arguments")
        return
    
    print(f"Starting transcript embedding process...")
    print(f"Using video list CSV: {args.video_list_csv}")
    print(f"Looking for transcripts in: {args.transcript_dir}")
    
    # Process videos based on CSV
    total_chunks, total_uploaded = process_videos_from_csv(
        args.video_list_csv,
        args.transcript_dir,
        args.skip_existing,
        args.limit
    )
    
    print(f"Processing complete! Generated {total_chunks} chunks and uploaded {total_uploaded} vectors.")

if __name__ == "__main__":
    main()