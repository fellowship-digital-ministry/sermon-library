import os
import glob
import json
from collections import defaultdict
from typing import Dict, List, Any

from .utils import BIBLE_REFERENCES_DIR, BibleReferenceStats

# Bible reference helper functions
def load_bible_references() -> Dict[str, List[Dict[str, Any]]]:
    """Load all Bible references from JSON files."""
    references = {}
    
    # Get all JSON files in the references directory
    reference_files = glob.glob(os.path.join(BIBLE_REFERENCES_DIR, "*.json"))
    
    for file_path in reference_files:
        try:
            book_name = os.path.basename(file_path).split('.')[0]
            with open(file_path, 'r', encoding='utf-8') as f:
                book_references = json.load(f)
                references[book_name] = book_references
        except Exception as e:
            print(f"Error loading reference file {file_path}: {str(e)}")
    
    return references

def get_bible_stats() -> BibleReferenceStats:
    """Generate statistics about Bible references."""
    all_references = load_bible_references()
    
    # Count references per book
    books_count = {book: len(references) for book, references in all_references.items()}
    
    # Count references per chapter
    chapters_count = defaultdict(lambda: defaultdict(int))
    for book, references in all_references.items():
        for ref in references:
            # Check if ref is a dictionary before using get()
            if isinstance(ref, dict) and ref.get('chapter'):
                chapter_key = f"{ref['chapter']}"
                chapters_count[book][chapter_key] += 1
            elif isinstance(ref, dict):
                # If there's no chapter key, count it as chapter "unknown"
                chapters_count[book]["unknown"] += 1
            else:
                # Handle the case where ref is not a dictionary
                print(f"Warning: Reference is not a dictionary: {ref}")
                # Count it as an "unknown" chapter
                chapters_count[book]["unknown"] += 1
    
    # Get top books by reference count
    top_books = [{"book": book, "count": count} 
                for book, count in sorted(books_count.items(), key=lambda x: x[1], reverse=True)[:10]]
    
    # Get top chapters by reference count
    chapter_flat_counts = []
    for book, chapters in chapters_count.items():
        for chapter, count in chapters.items():
            chapter_flat_counts.append({"book": book, "chapter": chapter, "count": count})
    
    top_chapters = sorted(chapter_flat_counts, key=lambda x: x["count"], reverse=True)[:10]
    
    # Count testament references
    old_testament_books = [
        "Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy", "Joshua", "Judges", "Ruth", 
        "1_Samuel", "2_Samuel", "1_Kings", "2_Kings", "1_Chronicles", "2_Chronicles", "Ezra", 
        "Nehemiah", "Esther", "Job", "Psalms", "Proverbs", "Ecclesiastes", "Song_of_Solomon", 
        "Isaiah", "Jeremiah", "Lamentations", "Ezekiel", "Daniel", "Hosea", "Joel", "Amos", 
        "Obadiah", "Jonah", "Micah", "Nahum", "Habakkuk", "Zephaniah", "Haggai", "Zechariah", "Malachi"
    ]
    
    old_testament_count = sum(len(all_references.get(book, [])) for book in old_testament_books)
    total_references = sum(len(refs) for refs in all_references.values())
    new_testament_count = total_references - old_testament_count
    
    return BibleReferenceStats(
        total_references=total_references,
        books_count=books_count,
        chapters_count=dict(chapters_count),
        top_books=top_books,
        top_chapters=top_chapters,
        old_testament_count=old_testament_count,
        new_testament_count=new_testament_count
    )


