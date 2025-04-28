"""
Utility functions for handling sermon metadata from JSON files.
Compatible with the sermon-library project structure.
"""

import os
import json
from typing import Dict, Any, Optional
import glob
from pathlib import Path
import re
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='metadata_utils.log'
)
logger = logging.getLogger('metadata_utils')

# Define multiple possible paths to the metadata directory
# This allows the code to work in different environments
POSSIBLE_METADATA_PATHS = [
    # Path relative to the current file (api directory)
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "transcription", "data", "metadata"),
    # Direct path if running from the project root
    os.path.join("transcription", "data", "metadata"),
    # GitHub Actions path
    os.path.join(".", "transcription", "data", "metadata"),
    # Absolute paths for different environments (update these as needed)
    "/home/runner/work/sermon-library/sermon-library/transcription/data/metadata"
]

# Cache to store loaded metadata
_metadata_cache: Dict[str, Dict[str, Any]] = {}
_cache_initialized = False
_metadata_dir = None

def _find_metadata_dir() -> Optional[str]:
    """Find the metadata directory by checking multiple possible paths."""
    # First check the predefined paths
    for path in POSSIBLE_METADATA_PATHS:
        if os.path.exists(path) and os.path.isdir(path):
            logger.info(f"Found metadata directory at: {path}")
            return path
    
    # If none of the predefined paths work, try to find it by searching
    # Start from the current directory and go up a few levels
    current_dir = os.path.dirname(os.path.abspath(__file__))
    for _ in range(4):  # Look up to 4 levels up
        # Search for a directory named "metadata" that contains .json files
        for root, dirs, files in os.walk(current_dir):
            if "metadata" in dirs:
                metadata_path = os.path.join(root, "metadata")
                # Check if this directory contains metadata.json files
                if any(f.endswith(".metadata.json") for f in os.listdir(metadata_path) if os.path.isfile(os.path.join(metadata_path, f))):
                    logger.info(f"Found metadata directory by searching: {metadata_path}")
                    return metadata_path
        # Move up one directory
        current_dir = os.path.dirname(current_dir)
    
    # Last resort: look for metadata files in any subdirectory
    logger.warning("Metadata directory not found in standard locations. Searching entire project...")
    try:
        # Get the project root (where .git is)
        project_root = current_dir
        while project_root and not os.path.exists(os.path.join(project_root, ".git")):
            parent = os.path.dirname(project_root)
            if parent == project_root:  # Reached filesystem root
                break
            project_root = parent
        
        if project_root:
            # Look for directories named "metadata" with JSON files
            for root, dirs, files in os.walk(project_root):
                if "metadata" in dirs:
                    metadata_path = os.path.join(root, "metadata")
                    json_files = [f for f in os.listdir(metadata_path) if f.endswith(".json") and os.path.isfile(os.path.join(metadata_path, f))]
                    if json_files:
                        logger.info(f"Found metadata directory in project search: {metadata_path}")
                        return metadata_path
    except Exception as e:
        logger.error(f"Error searching for metadata directory: {str(e)}")
    
    logger.error("Metadata directory not found")
    return None

def _initialize_metadata_cache() -> None:
    """Initialize the metadata cache by loading all metadata files."""
    global _cache_initialized, _metadata_dir
    
    if _cache_initialized:
        return
    
    # Find the metadata directory
    _metadata_dir = _find_metadata_dir()
    
    if not _metadata_dir:
        logger.warning("Metadata directory not found. Metadata features will be disabled.")
        _cache_initialized = True
        return
    
    logger.info(f"Loading metadata from directory: {_metadata_dir}")
    
    # Load all JSON files in the metadata directory
    metadata_files = glob.glob(os.path.join(_metadata_dir, "*.metadata.json"))
    
    # If no files found with *.metadata.json pattern, try *.json
    if not metadata_files:
        metadata_files = glob.glob(os.path.join(_metadata_dir, "*.json"))
    
    for file_path in metadata_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                
            # Extract video_id from the file name or from the JSON content
            video_id = metadata.get("video_id")
            
            # If not in the content, try to extract from filename
            if not video_id:
                file_name = os.path.basename(file_path)
                # Use regex to extract video ID from filename patterns like "*VIDEO_ID*.metadata.json"
                match = re.search(r'([^_\W]+[-\w]+)\.(?:metadata\.)?json$', file_name)
                if match:
                    video_id = match.group(1)
                else:
                    # Fallback to simply removing the extension
                    video_id = file_name.split(".")[0]
            
            if video_id:
                _metadata_cache[video_id] = metadata
                
        except Exception as e:
            logger.error(f"Error loading metadata from {file_path}: {str(e)}")
    
    logger.info(f"Loaded metadata for {len(_metadata_cache)} sermons")
    _cache_initialized = True

def get_sermon_metadata(video_id: str) -> Optional[Dict[str, Any]]:
    """
    Get metadata for a specific sermon by video_id.
    
    Args:
        video_id: The YouTube video ID
        
    Returns:
        Dictionary containing the sermon metadata or None if not found
    """
    if not _cache_initialized:
        _initialize_metadata_cache()
    
    # Check if we have the metadata in cache
    if video_id in _metadata_cache:
        return _metadata_cache[video_id]
    
    # If not in cache and metadata directory exists, try to load it directly
    if _metadata_dir:
        # Try several possible file patterns
        possible_patterns = [
            f"{video_id}.metadata.json",
            f"*{video_id}*.metadata.json",
            f"{video_id}.json",
            f"*{video_id}*.json"
        ]
        
        for pattern in possible_patterns:
            files = glob.glob(os.path.join(_metadata_dir, pattern))
            if files:
                try:
                    with open(files[0], 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    _metadata_cache[video_id] = metadata
                    return metadata
                except Exception as e:
                    logger.error(f"Error loading metadata from {files[0]}: {str(e)}")
    
    return None

def get_all_sermon_metadata() -> Dict[str, Dict[str, Any]]:
    """
    Get metadata for all sermons.
    
    Returns:
        Dictionary mapping video_id to sermon metadata
    """
    if not _cache_initialized:
        _initialize_metadata_cache()
    
    return _metadata_cache

def get_metadata_directory() -> Optional[str]:
    """
    Get the path to the metadata directory.
    
    Returns:
        Path to the metadata directory or None if not found
    """
    if not _cache_initialized:
        _initialize_metadata_cache()
    
    return _metadata_dir

def refresh_metadata_cache() -> int:
    """
    Refresh the metadata cache by reloading all metadata files.
    
    Returns:
        Number of metadata entries loaded
    """
    global _cache_initialized
    _cache_initialized = False
    _initialize_metadata_cache()
    return len(_metadata_cache)

def save_metadata(video_id: str, metadata: Dict[str, Any]) -> bool:
    """
    Save metadata for a sermon to a JSON file.
    
    Args:
        video_id: The YouTube video ID
        metadata: Dictionary containing metadata to save
        
    Returns:
        True if successful, False otherwise
    """
    if not _cache_initialized:
        _initialize_metadata_cache()
    
    if not _metadata_dir:
        logger.error("Cannot save metadata: Metadata directory not found")
        return False
    
    # Ensure the video_id is included in the metadata
    metadata["video_id"] = video_id
    
    # Construct the file path
    file_path = os.path.join(_metadata_dir, f"{video_id}.metadata.json")
    
    try:
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Write the metadata to the file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # Update the cache
        _metadata_cache[video_id] = metadata
        
        logger.info(f"Saved metadata for {video_id} to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving metadata for {video_id}: {str(e)}")
        return False

# Initialize the cache when the module is imported
if not _cache_initialized:
    _initialize_metadata_cache()