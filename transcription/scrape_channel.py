# scrape_channel.py
import yt_dlp
import csv
import os
import json
from datetime import datetime

def scrape_youtube_channel(channel_url, output_csv):
    """
    Scrape all videos from a YouTube channel and save metadata to CSV.
    
    Args:
        channel_url: URL of the YouTube channel
        output_csv: Path to save the CSV file
    """
    print(f"Scraping videos from: {channel_url}")
    
    # Configure yt-dlp options
    ydl_opts = {
        'ignoreerrors': True,  # Skip unavailable videos
        'extract_flat': True,  # Don't download videos, just get info
        'quiet': True,
        'no_warnings': True,
    }
    
    video_data = []
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # First, get the channel info
            channel_info = ydl.extract_info(channel_url, download=False)
            
            if 'entries' not in channel_info:
                print("No entries found in channel")
                return
            
            # Get number of videos for progress tracking
            total_videos = len(channel_info['entries'])
            print(f"Found {total_videos} videos in channel")
            
            # Process each video entry
            for i, entry in enumerate(channel_info['entries']):
                if i % 10 == 0:
                    print(f"Processing video {i+1}/{total_videos}")
                
                # Sometimes entries can be None for private/unavailable videos
                if entry is None:
                    continue
                
                video_id = entry.get('id')
                
                # Get detailed info for each video
                try:
                    # Configure detailed extraction
                    detailed_opts = {
                        'ignoreerrors': True,
                        'quiet': True,
                        'no_warnings': True,
                    }
                    
                    with yt_dlp.YoutubeDL(detailed_opts) as detailed_ydl:
                        video_url = f"https://www.youtube.com/watch?v={video_id}"
                        video_info = detailed_ydl.extract_info(video_url, download=False)
                        
                        # Extract metadata
                        video_data.append({
                            'video_id': video_id,
                            'title': video_info.get('title', ''),
                            'description': video_info.get('description', ''),
                            'publish_date': video_info.get('upload_date', ''),
                            'duration': video_info.get('duration', 0),
                            'view_count': video_info.get('view_count', 0),
                            'like_count': video_info.get('like_count', 0),
                            'url': video_url,
                            'thumbnail': video_info.get('thumbnail', '')
                        })
                except Exception as e:
                    print(f"Error processing video {video_id}: {str(e)}")
                    # Add basic info if detailed extraction fails
                    video_data.append({
                        'video_id': video_id,
                        'title': entry.get('title', ''),
                        'description': '',
                        'publish_date': '',
                        'duration': 0,
                        'view_count': 0,
                        'like_count': 0,
                        'url': f"https://www.youtube.com/watch?v={video_id}",
                        'thumbnail': ''
                    })
    
    except Exception as e:
        print(f"Error scraping channel: {str(e)}")
    
    # Save data to CSV
    if video_data:
        # Make sure the output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
        
        # Save to CSV
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['video_id', 'title', 'description', 'publish_date', 
                         'duration', 'view_count', 'like_count', 'url', 'thumbnail']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(video_data)
        
        print(f"Successfully saved {len(video_data)} videos to {output_csv}")
        
        # Also save raw data to JSON for backup
        json_path = output_csv.replace('.csv', '.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(video_data, f, indent=2, ensure_ascii=False)
        
        print(f"Backup data saved to {json_path}")
    else:
        print("No video data was collected")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Scrape videos from a YouTube channel")
    parser.add_argument("channel_url", help="URL of the YouTube channel")
    parser.add_argument("--output", "-o", default="data/video_list.csv", 
                        help="Output CSV file path (default: data/video_list.csv)")
    
    args = parser.parse_args()
    
    # Timestamp for backups
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create backup of existing file if it exists
    if os.path.exists(args.output):
        backup_path = f"{args.output}.{timestamp}.bak"
        os.rename(args.output, backup_path)
        print(f"Created backup of existing file: {backup_path}")
    
    # Run the scraper
    scrape_youtube_channel(args.channel_url, args.output)