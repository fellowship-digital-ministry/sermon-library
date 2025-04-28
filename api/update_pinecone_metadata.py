"""
Script to update metadata in Pinecone vectors from JSON metadata files.
Optimized for use with GitHub Actions workflow.

Can run in two modes:
1. Update all metadata (for initial sync)
2. Update only new/changed metadata (for regular updates)
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime, timedelta
from typing import Dict, Any, List, Set, Optional
import glob

# Add the current directory to the path so we can import metadata_utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from metadata_utils import get_all_sermon_metadata, get_metadata_directory
from pinecone import Pinecone

def find_new_or_modified_metadata(days: int = 7) -> Set[str]:
    """
    Find metadata files that are new or have been modified recently.
    
    Args:
        days: Number of days to look back
        
    Returns:
        Set of video IDs that have new or modified metadata
    """
    metadata_dir = get_metadata_directory()
    if not metadata_dir:
        print("Error: Metadata directory not found")
        return set()
    
    # Calculate the cutoff time
    cutoff_time = datetime.now() - timedelta(days=days)
    
    # Find all metadata files
    metadata_files = glob.glob(os.path.join(metadata_dir, "*.metadata.json"))
    
    # Filter to files modified after the cutoff time
    recent_files = []
    for file_path in metadata_files:
        mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
        if mod_time > cutoff_time:
            recent_files.append(file_path)
    
    # Extract video IDs from the filenames
    recent_video_ids = set()
    for file_path in recent_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            video_id = metadata.get("video_id")
            if video_id:
                recent_video_ids.add(video_id)
        except Exception as e:
            print(f"Error reading {file_path}: {str(e)}")
    
    return recent_video_ids

def update_pinecone_metadata(
    api_key: str, 
    index_name: str, 
    batch_size: int = 100, 
    dry_run: bool = False,
    only_recent: bool = False,
    days: int = 7,
    specific_ids: Optional[List[str]] = None
) -> None:
    """
    Update metadata in Pinecone with information from JSON metadata files.
    
    Args:
        api_key: Pinecone API key
        index_name: Name of the Pinecone index
        batch_size: Number of vectors to update in a single batch
        dry_run: If True, don't actually update Pinecone (just print what would be updated)
        only_recent: If True, only update metadata for recently modified files
        days: Number of days to look back for recent files
        specific_ids: List of specific video IDs to update (overrides only_recent)
    """
    # Initialize Pinecone client
    pc = Pinecone(api_key=api_key)
    
    # Connect to the index
    index = pc.Index(index_name)
    
    # Get all metadata from JSON files
    all_metadata = get_all_sermon_metadata()
    print(f"Loaded metadata for {len(all_metadata)} sermons from JSON files")
    
    if not all_metadata:
        print("No metadata found. Exiting.")
        return
    
    # Get index stats
    stats = index.describe_index_stats()
    total_vectors = stats.total_vector_count
    print(f"Found {total_vectors} vectors in Pinecone index '{index_name}'")
    
    # Determine which video IDs to process
    video_ids_to_process = set()
    
    if specific_ids:
        # Process only specified IDs
        video_ids_to_process = set(specific_ids)
        print(f"Will process {len(video_ids_to_process)} specified video IDs")
    elif only_recent:
        # Process only recently modified metadata
        video_ids_to_process = find_new_or_modified_metadata(days)
        print(f"Found {len(video_ids_to_process)} recently modified metadata files (last {days} days)")
    else:
        # Process all metadata
        video_ids_to_process = set(all_metadata.keys())
        print(f"Will process all {len(video_ids_to_process)} video IDs")
    
    # Filter metadata to only those we're processing
    metadata_to_process = {
        video_id: metadata for video_id, metadata in all_metadata.items()
        if video_id in video_ids_to_process
    }
    
    if not metadata_to_process:
        print("No metadata to process. Exiting.")
        return
    
    # Process each sermon
    updated_count = 0
    error_count = 0
    
    for video_id, metadata in metadata_to_process.items():
        try:
            print(f"Processing sermon: {video_id} - {metadata.get('title', 'Unknown Title')}")
            
            # Create a filter to find vectors with this video_id
            filter_dict = {"video_id": {"$eq": video_id}}
            
            # Use a generic query vector to find the vectors
            generic_vector = [0.1] * 1536  # Assuming 1536 dimensions
            
            # Find vectors for this sermon
            results = index.query(
                vector=generic_vector,
                top_k=1000,  # Get up to 1000 chunks per sermon
                include_metadata=True,
                filter=filter_dict
            )
            
            # If no vectors found, skip this sermon
            if not results.matches:
                print(f"  No vectors found for sermon {video_id}")
                continue
            
            print(f"  Found {len(results.matches)} vectors for sermon {video_id}")
            
            # Prepare batches of updates
            vectors_to_update = []
            
            for match in results.matches:
                # Get the vector ID
                vector_id = match.id
                
                # Get the existing metadata
                existing_metadata = match.metadata
                
                # Create a new metadata dict with both existing and new metadata
                new_metadata = {**existing_metadata}
                
                # Update with values from JSON metadata, but don't overwrite chunk-specific data
                for key, value in metadata.items():
                    if key not in ["text", "start_time", "end_time", "chunk_index", "segment_ids"]:
                        new_metadata[key] = value
                
                # Add to the list of vectors to update
                vectors_to_update.append((vector_id, new_metadata))
            
            # Process in batches
            for i in range(0, len(vectors_to_update), batch_size):
                batch = vectors_to_update[i:i+batch_size]
                
                # Print update details
                print(f"  {'Would update' if dry_run else 'Updating'} batch {i//batch_size + 1} ({len(batch)} vectors)")
                
                # Prepare the update batch
                update_batch = []
                for vector_id, metadata_update in batch:
                    update_batch.append({
                        "id": vector_id,
                        "metadata": metadata_update
                    })
                
                # Update Pinecone (unless dry run)
                if not dry_run:
                    index.update(vectors=update_batch)
                    # Sleep to avoid rate limits
                    time.sleep(0.5)
                
                updated_count += len(batch)
            
        except Exception as e:
            print(f"  Error processing sermon {video_id}: {str(e)}")
            error_count += 1
    
    print(f"\nSummary:")
    print(f"  Total sermons processed: {len(metadata_to_process)}")
    print(f"  Total vectors {'would be' if dry_run else ''} updated: {updated_count}")
    print(f"  Errors: {error_count}")
    
    if dry_run:
        print("\nThis was a dry run. No changes were made to Pinecone.")
        print("Run again with --dry-run=false to apply the changes.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update Pinecone metadata from JSON files")
    
    parser.add_argument("--api-key", help="Pinecone API key")
    parser.add_argument("--index-name", default="sermon-embeddings", help="Pinecone index name")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for updates")
    parser.add_argument("--dry-run", type=lambda x: x.lower() == 'true', default=False, 
                        help="Dry run mode (true/false)")
    parser.add_argument("--only-recent", action="store_true", 
                        help="Only update metadata for recently modified files")
    parser.add_argument("--days", type=int, default=7, 
                        help="Number of days to look back for recent files")
    parser.add_argument("--video-ids", 
                        help="Comma-separated list of specific video IDs to update")
    
    args = parser.parse_args()
    
    # Get API key from environment if not provided
    api_key = args.api_key or os.environ.get("PINECONE_API_KEY")
    if not api_key:
        print("Error: Pinecone API key not provided")
        sys.exit(1)
    
    # Parse video IDs if provided
    specific_ids = None
    if args.video_ids:
        specific_ids = [vid.strip() for vid in args.video_ids.split(",")]
    
    update_pinecone_metadata(
        api_key=api_key,
        index_name=args.index_name or os.environ.get("PINECONE_INDEX_NAME", "sermon-embeddings"),
        batch_size=args.batch_size,
        dry_run=args.dry_run,
        only_recent=args.only_recent,
        days=args.days,
        specific_ids=specific_ids
    )