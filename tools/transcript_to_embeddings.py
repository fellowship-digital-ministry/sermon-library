"""
Generate embeddings from sermon transcripts and store them in Pinecone.
This script processes JSON transcript files, chunks them appropriately,
and uploads the embeddings to Pinecone for semantic search.

Updated for Pinecone API v6.0+ without dotenv dependency
"""

import os
import json
import time
import argparse
from typing import List, Dict, Tuple, Optional
import logging
import uuid
from datetime import datetime
from tqdm import tqdm

import openai
from pinecone import Pinecone, ServerlessSpec, CloudProvider

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='embedding_processing.log'
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
    for index in pc.list_indexes():
        if index.name == PINECONE_INDEX_NAME:
            index_exists = True
            break
    
    # Create index if it doesn't exist
    if not index_exists:
        logger.info(f"Creating new Pinecone index: {PINECONE_INDEX_NAME}")
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
    
    # Connect to the index
    pinecone_index = pc.Index(PINECONE_INDEX_NAME)
    
    return openai_client, pinecone_index

def load_transcript(file_path: str) -> Dict:
    """Load and parse a transcript JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_video_metadata(transcript: Dict) -> Dict:
    """Extract metadata from the transcript."""
    video_id = transcript.get('video_id', '')
    
    # If we have metadata files, we could load additional information here
    metadata_path = os.path.join('transcription', 'data', 'metadata', f"{video_id}.json")
    metadata = {}
    
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    
    # Extract publish date if available
    publish_date = metadata.get('upload_date', datetime.now().strftime('%Y-%m-%d'))
    
    return {
        "video_id": video_id,
        "title": metadata.get('title', f"Sermon {video_id}"),
        "channel": metadata.get('channel', 'Unknown'),
        "publish_date": publish_date,
        "url": f"https://www.youtube.com/watch?v={video_id}"
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
        "start_time": segments[0]["start"],
        "end_time": segments[0]["end"],
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
        current_chunk["segment_ids"].append(segment["id"])
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
                
        except Exception as e:
            logger.error(f"Error generating embeddings for batch {i}: {str(e)}")
            # Insert empty embeddings as placeholders
            embeddings.extend([[] for _ in range(len(batch))])
    
    return embeddings

"""
Updated upload_to_pinecone function to fix the segment_ids metadata issue.
"""

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
            "segment_ids": segment_ids,  # Now a list of strings
            "word_count": len(chunk["text"].split())
        }
        
        vectors.append({
            "id": vector_id,
            "values": embedding,
            "metadata": vector_metadata
        })
    
    # Log the number of vectors to upload
    print(f"Preparing to upload {len(vectors)} vectors to Pinecone")
    logger.info(f"Preparing to upload {len(vectors)} vectors to Pinecone")
    
    # Upload vectors in batches
    uploaded = 0
    for i in range(0, len(vectors), BATCH_SIZE):
        batch = vectors[i:i + BATCH_SIZE]
        try:
            print(f"Uploading batch {i//BATCH_SIZE + 1} of {(len(vectors) + BATCH_SIZE - 1)//BATCH_SIZE}")
            index.upsert(vectors=batch)
            uploaded += len(batch)
            print(f"Successfully uploaded {uploaded}/{len(vectors)} vectors")
            
            # Avoid rate limiting
            if i + BATCH_SIZE < len(vectors):
                time.sleep(0.5)
                
        except Exception as e:
            error_message = f"Error uploading batch {i} to Pinecone: {str(e)}"
            print(f"ERROR: {error_message}")
            logger.error(error_message)
    
    return uploaded

def process_transcript_file(file_path: str, openai_client, pinecone_index) -> Tuple[int, int]:
    """
    Process a single transcript file.
    Returns the number of chunks and successfully uploaded vectors.
    """
    logger.info(f"Processing transcript: {file_path}")
    
    try:
        # Load and parse the transcript
        transcript = load_transcript(file_path)
        
        # Extract metadata
        metadata = extract_video_metadata(transcript)
        
        # Split transcript into chunks
        chunks = chunk_transcript(transcript)
        
        if not chunks:
            logger.warning(f"No chunks generated for {file_path}")
            return 0, 0
        
        # Get just the text from each chunk for embedding
        texts = [chunk["text"] for chunk in chunks]
        
        # Generate embeddings
        embeddings = create_embeddings(openai_client, texts)
        
        # Upload to Pinecone
        uploaded = upload_to_pinecone(pinecone_index, embeddings, chunks, metadata)
        
        logger.info(f"Successfully processed {file_path}: {len(chunks)} chunks, {uploaded} vectors uploaded")
        return len(chunks), uploaded
        
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        return 0, 0

def process_all_transcripts(transcript_dir: str, skip_existing: bool = True, limit: Optional[int] = None) -> Tuple[int, int]:
    """
    Process all transcript files in the specified directory.
    Returns the total number of chunks and successfully uploaded vectors.
    """
    try:
        # Initialize clients
        openai_client, pinecone_index = initialize_clients()
        
        # Get list of processed IDs if we're skipping existing
        processed_ids = set()
        if skip_existing:
            try:
                # Query for existing IDs in Pinecone
                stats = pinecone_index.describe_index_stats()
                # We could query for all vectors but that might be expensive
                # Instead, we'll use file modification time to decide what to process
                logger.info(f"Index stats: {stats}")
            except Exception as e:
                logger.warning(f"Could not fetch index stats: {str(e)}")
        
        # Get all transcript files
        file_paths = []
        for file in os.listdir(transcript_dir):
            if file.endswith('.json'):
                file_path = os.path.join(transcript_dir, file)
                file_paths.append(file_path)
        
        # Sort by modification time (newest first)
        file_paths.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        # Apply limit if specified
        if limit and limit > 0:
            file_paths = file_paths[:limit]
            logger.info(f"Processing limited to {limit} files")
        
        total_chunks = 0
        total_uploaded = 0
        
        # Process each file
        for file_path in tqdm(file_paths, desc="Processing transcripts"):
            video_id = os.path.basename(file_path).split('.')[0]
            
            # Skip if already processed and we're skipping existing
            if skip_existing:
                # Check file modification time
                mod_time = os.path.getmtime(file_path)
                # If file was modified in the last day, process it anyway
                # This allows for reprocessing recent files with updated algorithms
                is_recent = (time.time() - mod_time) < 86400  # 24 hours
                
                # If it's not recent and we've processed it before, skip
                if not is_recent and any(vector_id.startswith(video_id) for vector_id in processed_ids):
                    logger.info(f"Skipping already processed transcript: {video_id}")
                    continue
            
            chunks, uploaded = process_transcript_file(file_path, openai_client, pinecone_index)
            total_chunks += chunks
            total_uploaded += uploaded
        
        return total_chunks, total_uploaded
        
    except Exception as e:
        logger.error(f"Error in batch processing: {str(e)}")
        return 0, 0

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Generate embeddings from sermon transcripts")
    parser.add_argument("--transcript_dir", type=str, default="./transcription/data/transcripts",
                        help="Directory containing transcript JSON files")
    parser.add_argument("--skip_existing", action="store_true", default=True,
                        help="Skip transcripts that have already been processed")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit the number of transcripts to process")
    parser.add_argument("--api_key", type=str, default=None,
                        help="Pinecone API key (overrides environment variable)")
    args = parser.parse_args()
    
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
    
    # Process all transcripts
    total_chunks, total_uploaded = process_all_transcripts(
        args.transcript_dir, 
        args.skip_existing,
        args.limit
    )
    
    print(f"Processing complete! Generated {total_chunks} chunks and uploaded {total_uploaded} vectors.")

if __name__ == "__main__":
    main()