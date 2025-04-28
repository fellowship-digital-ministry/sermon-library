"""
Utility functions for handling sermon metadata from JSON files.
"""

import os
import json
from typing import Dict, Any, Optional
import glob
from pathlib import Path
import re

# Define multiple possible paths to the metadata directory
# This allows the code to work in different environments
POSSIBLE_METADATA_PATHS = [
    # Path relative to the current file (api directory)
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "transcription", "data", "metadata"),
    # Path for the structure you showed in your message
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sermon-library", "transcription", "data", "metadata"),
    # Direct path if running from the project root
    os.path.join("transcription", "data", "metadata"),
    # Fallback path with repo name
    os.path.join("sermon-library", "transcription", "data", "metadata")
]

# Cache to store loaded metadata
_metadata_cache: Dict[str, Dict[str, Any]] = {}
_cache_initialized = False
_metadata_dir = None

def _find_metadata_dir() -> Optional[str]:
    """Find the metadata directory by checking multiple possible paths."""
    for path in POSSIBLE_METADATA_PATHS:
        if os.path.exists(path) and os.path.isdir(path):
            return path
    
    # If none of the predefined paths work, try to find it by searching
    # Start from the current directory and go up a few levels
    current_dir = os.path.dirname(os.path.abspath(__file__))
    for _ in range(4):  # Look up to 4 levels up
        # Search for a directory named "metadata" that contains .json files
        for root, dirs, files in os.walk(current_dir):
            if "metadata" in dirs:
                metadata_path = os.path.join(root, "metadata")
                # Check if this directory contains .metadata.json files
                if any(f.endswith(".metadata.json") for f in os.listdir(metadata_path)):
                    return metadata_path
        # Move up one directory
        current_dir = os.path.dirname(current_dir)
    
    return None

def _initialize_metadata_cache() -> None:
    """Initialize the metadata cache by loading all metadata files."""
    global _cache_initialized, _metadata_dir
    
    if _cache_initialized:
        return
    
    # Find the metadata directory
    _metadata_dir = _find_metadata_dir()
    
    if not _metadata_dir:
        print("Warning: Metadata directory not found. Metadata features will be disabled.")
        _cache_initialized = True
        return
    
    print(f"Found metadata directory: {_metadata_dir}")
    
    # Load all JSON files in the metadata directory
    metadata_files = glob.glob(os.path.join(_metadata_dir, "*.metadata.json"))
    
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
                match = re.search(r'([^_\W]+[-\w]+)\.metadata\.json$', file_name)
                if match:
                    video_id = match.group(1)
                else:
                    # Fallback to simply removing the extension
                    video_id = file_name.split(".metadata.json")[0]
            
            if video_id:
                _metadata_cache[video_id] = metadata
                
        except Exception as e:
            print(f"Error loading metadata from {file_path}: {str(e)}")
    
    print(f"Loaded metadata for {len(_metadata_cache)} sermons")
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
            f"*{video_id}*.metadata.json"
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
                    print(f"Error loading metadata from {files[0]}: {str(e)}")
    
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