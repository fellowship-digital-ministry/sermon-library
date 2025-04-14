"""
JSON to SRT Converter for Sermon Transcripts

This script converts sermon transcript JSON files with timestamp segments into YouTube-compatible
SRT subtitle files that can be uploaded directly to sermon videos.

Purpose:
--------
The primary purpose of this tool is to enhance sermon accessibility by creating
subtitle files from existing transcripts. SRT files can be uploaded to YouTube
to provide closed captions, making sermons accessible to:
- Deaf or hard-of-hearing viewers
- Non-native English speakers
- Those watching in sound-sensitive environments
- People who prefer reading along with spoken content

Features:
---------
- Converts single JSON files or entire directories of transcripts
- Preserves precise timestamp information for accurate subtitle synchronization
- Formats output according to standard SRT specifications
- Handles various JSON transcript structures
- Provides detailed logging of conversion process

Usage:
------
1. Single file conversion:
   python json_to_srt.py --input data/transcripts/sermon_123.json --output data/subtitles

2. Batch conversion of all transcripts:
   python json_to_srt.py --input data/transcripts --output data/subtitles

SRT Format:
-----------
The SRT (SubRip Text) format consists of:
1. A sequence number
2. Start and end timestamps in HH:MM:SS,mmm format
3. The subtitle text
4. A blank line to separate entries

Once generated, SRT files can be uploaded to YouTube through:
YouTube Studio > Video > Subtitles > Add > Upload File > With timing

This tool is part of the Fellowship Baptist Church Sermon Library project.
"""
import json
import os
import argparse
import glob
from datetime import datetime

def format_timestamp_srt(seconds):
    """
    Format seconds as SRT timestamp: HH:MM:SS,mmm
    """
    hours = int(seconds / 3600)
    minutes = int((seconds % 3600) / 60)
    seconds_part = seconds % 60
    milliseconds = int((seconds_part - int(seconds_part)) * 1000)
    
    return f"{hours:02d}:{minutes:02d}:{int(seconds_part):02d},{milliseconds:03d}"

def convert_json_to_srt(json_file, output_dir=None):
    """
    Convert a JSON transcript file to SRT format.
    
    Args:
        json_file: Path to the JSON transcript file
        output_dir: Directory to save the SRT file (default: same directory as JSON)
    
    Returns:
        Path to the generated SRT file
    """
    try:
        # Determine output directory
        if output_dir is None:
            output_dir = os.path.dirname(json_file)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Read JSON file
        with open(json_file, 'r', encoding='utf-8') as f:
            transcript_data = json.load(f)
        
        # Extract video_id from JSON or filename
        if 'video_id' in transcript_data:
            video_id = transcript_data['video_id']
        else:
            video_id = os.path.splitext(os.path.basename(json_file))[0]
        
        # Create SRT filename
        srt_file = os.path.join(output_dir, f"{video_id}.srt")
        
        # Get segments from JSON
        segments = transcript_data.get("segments", [])
        if not segments:
            print(f"No segments found in {json_file}")
            return None
        
        # Write SRT file
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
        
        print(f"Successfully converted {json_file} to {srt_file}")
        return srt_file
    
    except Exception as e:
        print(f"Error converting {json_file}: {str(e)}")
        return None

def batch_convert(input_dir, output_dir=None):
    """
    Convert all JSON transcripts in a directory to SRT files.
    
    Args:
        input_dir: Directory containing JSON transcript files
        output_dir: Directory to save SRT files (default: same as input_dir)
    
    Returns:
        Number of files successfully converted
    """
    if output_dir is None:
        output_dir = input_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all JSON files in the input directory
    json_files = glob.glob(os.path.join(input_dir, "*.json"))
    
    success_count = 0
    
    print(f"Found {len(json_files)} JSON files to convert")
    
    for json_file in json_files:
        if convert_json_to_srt(json_file, output_dir):
            success_count += 1
    
    print(f"Successfully converted {success_count} out of {len(json_files)} files")
    return success_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert JSON transcripts to SRT files")
    parser.add_argument("--input", "-i", help="Input JSON file or directory", required=True)
    parser.add_argument("--output", "-o", help="Output directory for SRT files")
    
    args = parser.parse_args()
    
    if os.path.isdir(args.input):
        batch_convert(args.input, args.output)
    else:
        convert_json_to_srt(args.input, args.output)