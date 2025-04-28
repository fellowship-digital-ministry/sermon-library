"""
Test script for the metadata utilities.
Run this script to verify that the metadata files can be loaded correctly.
"""

import os
import sys
from pprint import pprint

# Add the current directory to the path so we can import metadata_utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from metadata_utils import get_all_sermon_metadata, get_sermon_metadata, get_metadata_directory
    
    # Print the metadata directory
    metadata_dir = get_metadata_directory()
    print(f"Metadata directory: {metadata_dir}")
    
    # Get all metadata
    all_metadata = get_all_sermon_metadata()
    print(f"Found metadata for {len(all_metadata)} sermons")
    
    # Print a sample of metadata
    if all_metadata:
        sample_video_id = next(iter(all_metadata.keys()))
        print("\nSample metadata for video ID:", sample_video_id)
        pprint(get_sermon_metadata(sample_video_id))
    else:
        print("\nNo metadata found.")
    
except Exception as e:
    print(f"Error: {str(e)}")
    import traceback
    traceback.print_exc()