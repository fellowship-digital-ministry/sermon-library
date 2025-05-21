import os
import json
import re
import glob
import time
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# List of KJV Bible books for validation
BIBLE_BOOKS = [
    "Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy", "Joshua", "Judges", "Ruth",
    "1 Samuel", "2 Samuel", "1 Kings", "2 Kings", "1 Chronicles", "2 Chronicles", "Ezra", "Nehemiah",
    "Esther", "Job", "Psalms", "Psalm", "Proverbs", "Ecclesiastes", "Song of Solomon", "Isaiah", "Jeremiah",
    "Lamentations", "Ezekiel", "Daniel", "Hosea", "Joel", "Amos", "Obadiah", "Jonah", "Micah", "Nahum",
    "Habakkuk", "Zephaniah", "Haggai", "Zechariah", "Malachi", "Matthew", "Mark", "Luke", "John",
    "Acts", "Romans", "1 Corinthians", "2 Corinthians", "Galatians", "Ephesians", "Philippians",
    "Colossians", "1 Thessalonians", "2 Thessalonians", "1 Timothy", "2 Timothy", "Titus", "Philemon",
    "Hebrews", "James", "1 Peter", "2 Peter", "1 John", "2 John", "3 John", "Jude", "Revelation"
]

class BibleReferenceExtractor:
    """
    A class to extract KJV Bible references from sermon transcripts using OpenAI's GPT-4o model.
    Implements user-friendly processing with progress bars, error handling, and efficient processing.
    """
    
    def __init__(self, api_key, input_dir, output_dir, chunk_size=1000, batch_size=10, max_workers=4, force_reprocess=False):
        """
        Initialize the BibleReferenceExtractor with configuration parameters.
        
        Args:
            api_key (str): OpenAI API key
            input_dir (str): Directory containing sermon transcript JSON files
            output_dir (str): Directory to save the output JSON files by Bible book
            chunk_size (int): Number of characters in each chunk for processing
            batch_size (int): Number of chunks to process in a single batch
            max_workers (int): Maximum number of parallel workers
            force_reprocess (bool): Whether to reprocess files that have already been processed
        """
        # Use the provided API key or get from environment if None
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("No OpenAI API key provided and OPENAI_API_KEY environment variable not set")
        
        self.client = OpenAI(api_key=api_key)
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.force_reprocess = force_reprocess
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Process tracking file path
        self.processed_files_path = os.path.join(output_dir, "processed_files.json")
        
        # List of KJV Bible books for validation
        self.bible_books = BIBLE_BOOKS
        
        # Include common abbreviations
        self.bible_book_abbreviations = {
            "Gen": "Genesis", "Exo": "Exodus", "Lev": "Leviticus", "Num": "Numbers", "Deut": "Deuteronomy",
            "Josh": "Joshua", "Judg": "Judges", "1 Sam": "1 Samuel", "2 Sam": "2 Samuel", "1 Kgs": "1 Kings",
            "2 Kgs": "2 Kings", "1 Chr": "1 Chronicles", "2 Chr": "2 Chronicles", "Neh": "Nehemiah", 
            "Psa": "Psalms", "Ps": "Psalms", "Prov": "Proverbs", "Eccl": "Ecclesiastes", "Song": "Song of Solomon",
            "Isa": "Isaiah", "Jer": "Jeremiah", "Lam": "Lamentations", "Ezek": "Ezekiel", "Dan": "Daniel",
            "Hos": "Hosea", "Zech": "Zechariah", "Mal": "Malachi", "Matt": "Matthew", "Rom": "Romans",
            "1 Cor": "1 Corinthians", "2 Cor": "2 Corinthians", "Gal": "Galatians", "Eph": "Ephesians",
            "Phil": "Philippians", "Col": "Colossians", "1 Thess": "1 Thessalonians", "2 Thess": "2 Thessalonians",
            "1 Tim": "1 Timothy", "2 Tim": "2 Timothy", "Heb": "Hebrews", "Jas": "James", "1 Pet": "1 Peter",
            "2 Pet": "2 Peter", "1 Jn": "1 John", "2 Jn": "2 John", "3 Jn": "3 John", "Rev": "Revelation"
        }
        
        # Initialize the reference storage dictionary
        self.references_by_book = {}
        
        # Initialize processed files tracking
        self.processed_files = self._load_processed_files()
        
        # Load existing references
        self.load_existing_references()
    
    def _load_processed_files(self):
        """Load the list of already processed files"""
        if os.path.exists(self.processed_files_path):
            try:
                with open(self.processed_files_path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"Error loading processed files tracking, initializing empty")
                return {}
        return {}
    
    def _save_processed_files(self):
        """Save the list of processed files"""
        with open(self.processed_files_path, 'w') as f:
            json.dump(self.processed_files, f, indent=2)
            
    def _mark_file_as_processed(self, file_path, num_references=0):
        """Mark a file as processed with a timestamp and reference count"""
        rel_path = os.path.relpath(file_path, self.input_dir)
        self.processed_files[rel_path] = {
            "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "references_found": num_references
        }
        self._save_processed_files()
        
        # List of KJV Bible books for validation
        self.bible_books = BIBLE_BOOKS
        
        # Include common abbreviations
        self.bible_book_abbreviations = {
            "Gen": "Genesis", "Exo": "Exodus", "Lev": "Leviticus", "Num": "Numbers", "Deut": "Deuteronomy",
            "Josh": "Joshua", "Judg": "Judges", "1 Sam": "1 Samuel", "2 Sam": "2 Samuel", "1 Kgs": "1 Kings",
            "2 Kgs": "2 Kings", "1 Chr": "1 Chronicles", "2 Chr": "2 Chronicles", "Neh": "Nehemiah", 
            "Psa": "Psalms", "Ps": "Psalms", "Prov": "Proverbs", "Eccl": "Ecclesiastes", "Song": "Song of Solomon",
            "Isa": "Isaiah", "Jer": "Jeremiah", "Lam": "Lamentations", "Ezek": "Ezekiel", "Dan": "Daniel",
            "Hos": "Hosea", "Zech": "Zechariah", "Mal": "Malachi", "Matt": "Matthew", "Rom": "Romans",
            "1 Cor": "1 Corinthians", "2 Cor": "2 Corinthians", "Gal": "Galatians", "Eph": "Ephesians",
            "Phil": "Philippians", "Col": "Colossians", "1 Thess": "1 Thessalonians", "2 Thess": "2 Thessalonians",
            "1 Tim": "1 Timothy", "2 Tim": "2 Timothy", "Heb": "Hebrews", "Jas": "James", "1 Pet": "1 Peter",
            "2 Pet": "2 Peter", "1 Jn": "1 John", "2 Jn": "2 John", "3 Jn": "3 John", "Rev": "Revelation"
        }
        
        # Initialize the reference storage dictionary
        self.references_by_book = {}
        self.load_existing_references()
    
    def load_existing_references(self):
        """Load existing reference JSON files if they exist"""
        for book in self.bible_books:
            book_file = os.path.join(self.output_dir, f"{book.replace(' ', '_')}.json")
            if os.path.exists(book_file):
                try:
                    with open(book_file, 'r') as f:
                        self.references_by_book[book] = json.load(f)
                        print(f"Loaded existing references for {book}")
                except json.JSONDecodeError:
                    print(f"Error loading {book_file}, initializing empty")
                    self.references_by_book[book] = []
            else:
                self.references_by_book[book] = []
    
    def save_references(self):
        """Save references to JSON files organized by Bible book"""
        for book, references in self.references_by_book.items():
            if references:  # Only save if there are references
                book_filename = book.replace(' ', '_')
                output_file = os.path.join(self.output_dir, f"{book_filename}.json")
                
                with open(output_file, 'w') as f:
                    json.dump(references, f, indent=2)
                print(f"Saved {len(references)} references to {output_file}")
    
    def chunk_transcript(self, transcript_json):
        """
        Split transcript into manageable chunks with context overlap.
        Each chunk maintains the original timing information.
        
        Args:
            transcript_json (dict): The loaded transcript JSON data
            
        Returns:
            list: List of dictionaries with chunk text and timing information
        """
        chunks = []
        
        if "segments" not in transcript_json:
            print(f"Warning: Missing 'segments' field in transcript {transcript_json.get('video_id', 'unknown')}")
            return chunks
        
        segments = transcript_json["segments"]
        if not segments:
            return chunks
            
        current_chunk = {
            "video_id": transcript_json["video_id"],
            "text": "",
            "start": segments[0]["start"],
            "end": segments[0]["end"]
        }
        
        for segment in segments:
            # If adding this segment would exceed chunk size, finish current chunk
            if len(current_chunk["text"]) + len(segment["text"]) > self.chunk_size:
                # Ensure the chunk is meaningful
                if len(current_chunk["text"]) > 100:
                    chunks.append(current_chunk)
                
                # Start a new chunk
                current_chunk = {
                    "video_id": transcript_json["video_id"],
                    "text": segment["text"],
                    "start": segment["start"],
                    "end": segment["end"]
                }
            else:
                # Add segment to current chunk
                current_chunk["text"] += segment["text"]
                current_chunk["end"] = segment["end"]
        
        # Add the last chunk if it has content
        if current_chunk["text"] and len(current_chunk["text"]) > 100:
            chunks.append(current_chunk)
            
        return chunks
    
    def extract_references_with_gpt(self, chunk):
        """
        Use OpenAI's GPT-4o to identify KJV Bible references in a chunk of text.
        
        Args:
            chunk (dict): Dictionary containing chunk text and timing info
            
        Returns:
            list: List of identified Bible references with timestamps
        """
        # Print what we're processing for more visibility
        chunk_text = chunk.get("text", "")
        if not chunk_text:
            print("Warning: Empty chunk text")
            return []
            
        if len(chunk_text) < 100:
            sample_text = chunk_text
        else:
            sample_text = chunk_text[:100] + "..."
        print(f"Processing chunk: {sample_text}")
        
        # Notice the word "JSON" is included in the system message to satisfy the API requirement
        system_message = "You are a biblical scholar identifying Bible references in sermon transcripts. Return your response as a valid JSON object."
        
        prompt = """
        Identify King James Version Bible references in this sermon transcript.
        
        Format your response as a JSON object EXACTLY like this:
        {{"references": [
            {{"book": "John", "chapter": 3, "verse": 16, "reference_text": "John 3:16", "context": "...", "is_implicit": false}},
            {{"book": "Romans", "chapter": 5, "verse": 8, "reference_text": "Romans 5:8", "context": "...", "is_implicit": false}}
        ]}}
        
        If no references are found, return:
        {{"references": []}}
        
        Here's the transcript:
        {0}
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt.format(chunk_text)}
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
                max_tokens=1000
            )
            
            # Get the raw response content
            raw_content = response.choices[0].message.content
            print(f"Raw response content (first 100 chars): {raw_content[:100]}")
            
            try:
                # Parse the JSON response
                result = json.loads(raw_content)
                
                # Basic validation
                if not isinstance(result, dict):
                    print(f"Result is not a dictionary: {type(result)}")
                    return []
                
                if "references" not in result:
                    print(f"No 'references' key in the result: {list(result.keys())}")
                    return []
                
                references = result["references"]
                if not isinstance(references, list):
                    print(f"'references' is not a list: {type(references)}")
                    return []
                
                # Process each reference
                processed_refs = []
                for i, ref in enumerate(references):
                    try:
                        # Skip invalid references
                        if not isinstance(ref, dict):
                            print(f"Reference #{i} is not a dictionary: {ref}")
                            continue
                        
                        # Get the book and skip if missing
                        book = ref.get("book")
                        if not book:
                            print(f"Reference #{i} missing 'book': {ref}")
                            continue
                        
                        # Create a new reference object with all fields
                        new_ref = {
                            "book": str(book),  # Ensure it's a string
                            "chapter": ref.get("chapter"),
                            "verse": ref.get("verse"),
                            "reference_text": ref.get("reference_text", ""),
                            "context": ref.get("context", ""),
                            "is_implicit": ref.get("is_implicit", False),
                            "video_id": chunk.get("video_id"),
                            "start_time": chunk.get("start"),
                            "end_time": chunk.get("end")
                        }
                        
                        # Standardize book names
                        if book in self.bible_book_abbreviations:
                            new_ref["book"] = self.bible_book_abbreviations[book]
                        elif book.startswith("First "):
                            new_ref["book"] = book.replace("First ", "1 ")
                        elif book.startswith("Second "):
                            new_ref["book"] = book.replace("Second ", "2 ")
                        elif book.startswith("Third "):
                            new_ref["book"] = book.replace("Third ", "3 ")
                        
                        processed_refs.append(new_ref)
                    except Exception as e:
                        print(f"Error processing reference #{i}: {e}")
                
                return processed_refs
                
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                print(f"Raw content: {raw_content}")
                return []
                
        except Exception as e:
            print(f"API error: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def process_chunks_batch(self, chunks):
        """Process a batch of chunks in parallel"""
        all_references = []
        
        # Instead of using ThreadPoolExecutor, process chunks sequentially for debugging
        for chunk in tqdm(chunks, desc="Processing chunks"):
            try:
                references = self.extract_references_with_gpt(chunk)
                all_references.extend(references)
            except Exception as e:
                print(f"Error in process_chunks_batch: {e}")
                import traceback
                traceback.print_exc()
        
        return all_references
    
    def organize_references_by_book(self, references):
        """Organize references by Bible book"""
        for ref in references:
            book = ref.get("book")
            if book and book in self.bible_books:
                if book not in self.references_by_book:
                    self.references_by_book[book] = []
                self.references_by_book[book].append(ref)
            elif book and book in self.bible_book_abbreviations:
                standardized_book = self.bible_book_abbreviations[book]
                if standardized_book not in self.references_by_book:
                    self.references_by_book[standardized_book] = []
                ref["book"] = standardized_book
                self.references_by_book[standardized_book].append(ref)
    
    def process_transcript_file(self, file_path):
        """Process a single transcript file"""
        # Check if file has already been processed
        rel_path = os.path.relpath(file_path, self.input_dir)
        if not self.force_reprocess and rel_path in self.processed_files:
            print(f"Skipping already processed file: {rel_path}")
            print(f"  Processed on: {self.processed_files[rel_path]['processed_at']}")
            print(f"  References found: {self.processed_files[rel_path]['references_found']}")
            return []
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                transcript_json = json.load(f)
            
            video_id = transcript_json.get("video_id", "unknown")
            print(f"Processing transcript: {video_id}")
            
            # Split transcript into chunks
            chunks = self.chunk_transcript(transcript_json)
            print(f"Split into {len(chunks)} chunks")
            
            # Process chunks in batches
            all_references = []
            for i in range(0, len(chunks), self.batch_size):
                batch = chunks[i:i + self.batch_size]
                batch_references = self.process_chunks_batch(batch)
                all_references.extend(batch_references)
                print(f"Processed batch {i//self.batch_size + 1}/{(len(chunks) + self.batch_size - 1)//self.batch_size}")
                
                # Save references after each batch to avoid losing progress
                self.organize_references_by_book(batch_references)
                self.save_references()
                
                # Throttle to avoid rate limits
                time.sleep(1)
            
            print(f"Found {len(all_references)} references in {video_id}")
            
            # Mark file as processed
            self._mark_file_as_processed(file_path, len(all_references))
            
            return all_references
            
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            return []
    
    def process_all_transcripts(self, limit=None):
        """Process all transcript files in the input directory"""
        file_pattern = os.path.join(self.input_dir, "*.json")
        transcript_files = glob.glob(file_pattern)
        
        if not transcript_files:
            print(f"No transcript files found in {self.input_dir}")
            return
        
        print(f"Found {len(transcript_files)} transcript files")
        
        # Filter out already processed files unless forced to reprocess
        if not self.force_reprocess:
            unprocessed_files = []
            skipped_files = 0
            
            for file_path in transcript_files:
                rel_path = os.path.relpath(file_path, self.input_dir)
                if rel_path in self.processed_files:
                    skipped_files += 1
                else:
                    unprocessed_files.append(file_path)
            
            if skipped_files > 0:
                print(f"Skipping {skipped_files} already processed files")
                print(f"Found {len(unprocessed_files)} unprocessed files")
            
            transcript_files = unprocessed_files
            
            if not transcript_files:
                print("All files have already been processed")
                print("Use --force-reprocess to reprocess them anyway")
                return
        
        # Apply limit if specified
        if limit is not None and limit < len(transcript_files):
            print(f"Limiting to {limit} files in this run (out of {len(transcript_files)} available)")
            transcript_files = transcript_files[:limit]
        
        for file_path in tqdm(transcript_files, desc="Processing files"):
            self.process_transcript_file(file_path)
            # Save after each file
            self.save_references()


def main():
    """Main function to run the Bible reference extractor"""
    import argparse
    
    # Get API key from environment variable
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set it using: export OPENAI_API_KEY='your-key' (Linux/Mac)")
        print("Or: set OPENAI_API_KEY=your-key (Windows)")
        return
    
    parser = argparse.ArgumentParser(description="Extract KJV Bible references from sermon transcripts")
    parser.add_argument("--input-dir", default="./transcripts", help="Directory containing sermon transcript JSON files")
    parser.add_argument("--output-dir", default="./bible_references", help="Directory to save output JSON files")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Characters per chunk")
    parser.add_argument("--batch-size", type=int, default=5, help="Chunks to process in a batch")
    parser.add_argument("--max-workers", type=int, default=4, help="Maximum parallel workers")
    parser.add_argument("--files", nargs="*", help="Specific transcript files to process (if not specified, all files in input-dir will be processed)")
    parser.add_argument("--force-reprocess", action="store_true", help="Force reprocessing of already processed files")
    parser.add_argument("--list-processed", action="store_true", help="Only list already processed files without processing anything")
    parser.add_argument("--limit", type=int, help="Limit the number of files to process in a single run")
    
    args = parser.parse_args()
    
    print("Bible Reference Extractor")
    print("========================")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    
    # Create extractor instance
    extractor = BibleReferenceExtractor(
        api_key=api_key,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        force_reprocess=args.force_reprocess
    )
    
    # Handle --list-processed flag
    if args.list_processed:
        if not extractor.processed_files:
            print("No files have been processed yet.")
        else:
            print(f"Processed files ({len(extractor.processed_files)}):")
            for rel_path, info in extractor.processed_files.items():
                print(f"  {rel_path}")
                print(f"    Processed on: {info['processed_at']}")
                print(f"    References found: {info['references_found']}")
        return
    
    print(f"Chunk size: {args.chunk_size} characters")
    print(f"Batch size: {args.batch_size} chunks")
    print(f"Max workers: {args.max_workers}")
    
    if args.force_reprocess:
        print("NOTICE: Forcing reprocessing of all files, even if already processed")
    
    if args.limit:
        print(f"Limiting to {args.limit} files in this run")
    
    if args.files:
        print(f"Processing specific files: {', '.join(args.files)}")
    else:
        print("Processing all files in input directory")
    print("========================")
    
    if args.files:
        # Process only the specified files
        files_to_process = args.files
        if args.limit and args.limit < len(files_to_process):
            files_to_process = files_to_process[:args.limit]
            print(f"Limiting to first {args.limit} specified files")
            
        for filename in files_to_process:
            file_path = os.path.join(args.input_dir, filename)
            if os.path.exists(file_path):
                extractor.process_transcript_file(file_path)
            else:
                print(f"Warning: File not found: {file_path}")
        extractor.save_references()
    else:
        # Process all files in the directory with optional limit
        if args.limit:
            extractor.process_all_transcripts(limit=args.limit)
        else:
            extractor.process_all_transcripts()
    
    print("Processing complete!")


if __name__ == "__main__":
    main()