import os
import time
import glob
import json
from typing import List, Dict, Optional, Any, Union
from datetime import datetime
from collections import Counter, defaultdict

from fastapi import FastAPI, HTTPException, Query, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import openai
# Updated Pinecone import for version 6.0.2
from pinecone import Pinecone

# Configure API keys and settings directly from environment variables
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT", "us-east-1")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "sermon-embeddings")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
SEARCH_TOP_K = int(os.environ.get("SEARCH_TOP_K", "5"))
COMPLETION_MODEL = os.environ.get("COMPLETION_MODEL", "gpt-4o")
TRANSLATION_MODEL = os.environ.get("TRANSLATION_MODEL", "gpt-4o") # Smaller model for translations

# Path to metadata directory
METADATA_DIR = os.environ.get("METADATA_DIR", "./transcription/data/metadata")
SUBTITLES_DIR = os.environ.get("SUBTITLES_DIR", "./transcription/data/subtitles")
# Add Bible references directory path
BIBLE_REFERENCES_DIR = os.environ.get("BIBLE_REFERENCES_DIR", "./transcription/data/bible_references")

# Check for required environment variables
if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY environment variable")
if not PINECONE_API_KEY:
    raise ValueError("Missing PINECONE_API_KEY environment variable")

# Initialize clients
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Initialize Pinecone with version 6.0.2 API
pc = Pinecone(api_key=PINECONE_API_KEY)

# Connect to the index
pinecone_index = pc.Index(PINECONE_INDEX_NAME)

# Initialize FastAPI app
app = FastAPI(
    title="Sermon Search API",
    description="API for searching sermon transcripts and generating answers from the content",
    version="1.0.0"
)

# Add CORS middleware to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://fellowship-digital-ministry.github.io", "*"],  # Include your GitHub Pages domain explicitly
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define models
class SearchResult(BaseModel):
    video_id: str
    title: str
    url: str
    text: str
    start_time: float
    end_time: float
    similarity: float
    chunk_index: int
    segment_ids: List[str] = []
    publish_date: Optional[int] = None

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total_results: int
    processing_time: float

class AnswerRequest(BaseModel):
    query: str = Field(..., description="The question to answer based on sermon content")
    top_k: int = Field(5, description="Number of search results to consider")
    include_sources: bool = Field(True, description="Whether to include source information in the response")
    language: str = Field("en", description="Language for the response (en, es, zh)")
    # New fields for title and date filtering
    title: Optional[str] = Field(None, description="Filter by sermon title")
    date: Optional[str] = Field(None, description="Filter by sermon date (YYYY-MM-DD)")
    preacher: Optional[str] = Field(None, description="Filter by preacher name")

class AnswerResponse(BaseModel):
    query: str
    answer: str
    sources: List[SearchResult] = []
    processing_time: float

# Bible reference models
class BibleBook(BaseModel):
    book: str
    count: int
    references: List[Dict[str, Any]] = []
    
class BibleReference(BaseModel):
    book: str
    chapter: Optional[int]
    verse: Optional[int]
    reference_text: str
    context: str
    is_implicit: bool
    video_id: str
    start_time: float
    end_time: float

class BibleReferenceStats(BaseModel):
    total_references: int
    books_count: Dict[str, int]
    chapters_count: Dict[str, Dict[str, int]]
    top_books: List[Dict[str, Any]]
    top_chapters: List[Dict[str, Any]]
    old_testament_count: int
    new_testament_count: int

# New model for sermon metadata
class SermonMetadata(BaseModel):
    video_id: str
    title: str
    publish_date: Optional[str] = None
    preacher: Optional[str] = None
    url: str

# Helper function to get full language name
def get_language_name(lang_code):
    """Return the full language name from a language code."""
    language_names = {
        "en": "English",
        "es": "Spanish",
        "zh": "Chinese"
    }
    return language_names.get(lang_code, "Unknown")

# Helper function for translation
async def translate_text(text, source_lang, target_lang="en"):
    """
    Translate text using GPT model
    source_lang: The source language code (e.g., 'zh', 'es')
    target_lang: The target language code (default: 'en')
    """
    if not text or source_lang == target_lang:
        return text  # No need to translate if languages match

    try:
        # Construct the prompt based on direction of translation
        if target_lang == "en":
            prompt = f"Translate the following text from {get_language_name(source_lang)} to English. Preserve all information accurately: \n\n{text}"
        else:
            prompt = f"Translate the following text from English to {get_language_name(target_lang)}. Preserve all information accurately: \n\n{text}"
        
        response = openai_client.chat.completions.create(
            model=TRANSLATION_MODEL,  # Use smaller model for translation
            messages=[
                {"role": "system", "content": "You are a professional translator."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1024
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Translation error: {str(e)}")
        # If translation fails, return original text
        return text

# Helper function to extract language from header
def get_language(accept_language: Optional[str] = Header(None, include_in_schema=False)) -> str:
    """Extract and validate the preferred language from the Accept-Language header."""
    if not accept_language:
        return "en"
    
    # Simple parsing of Accept-Language header
    languages = [lang.split(";")[0].strip() for lang in accept_language.split(",")]
    
    # Check if any of our supported languages are requested
    for lang in languages:
        if lang.startswith("es"):
            return "es"
        elif lang.startswith("zh"):
            return "zh"
    
    # Default to English
    return "en"

# Helper function to load metadata
def load_metadata(video_id):
    """Load metadata for a video from its JSON file."""
    try:
        metadata_file = os.path.join(METADATA_DIR, f"{video_id}_metadata.json")
        with open(metadata_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            "video_id": video_id,
            "title": f"Unknown Sermon ({video_id})",
            "publish_date": None
        }
    except json.JSONDecodeError:
        return {
            "video_id": video_id,
            "title": f"Error Loading Metadata ({video_id})",
            "publish_date": None
        }

# Functions
def generate_embedding(text: str) -> List[float]:
    """Generate an embedding for the given text."""
    try:
        response = openai_client.embeddings.create(
            input=text,
            model=EMBEDDING_MODEL
        )
        return response.data[0].embedding
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating embedding: {str(e)}")

def format_time(seconds: float) -> str:
    """Format time in seconds to MM:SS format."""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes}:{seconds:02d}"

def get_youtube_timestamp_url(video_id: str, seconds: float) -> str:
    """Generate a YouTube URL with a timestamp."""
    return f"https://www.youtube.com/watch?v={video_id}&t={int(seconds)}"

# Modified to include sermon metadata in the context
def generate_ai_answer(query: str, search_results: List[SearchResult], language: str = "en") -> str:
    """Generate an AI answer based on the search results in the specified language."""
    # Prepare the context from search results with enhanced metadata
    context = "\n\n".join([
        f"Segment {i+1} (Sermon: \"{result.title}\", Date: {result.publish_date}, Time: {format_time(result.start_time)} - {format_time(result.end_time)}):\n{result.text}"
        for i, result in enumerate(search_results)
    ])
    
    # Set system message based on language
    if language == "es":
        system_message = "Eres un asistente que ayuda a los usuarios a entender el contenido de sermones. Responde en español."
    elif language == "zh":
        system_message = "你是一个帮助用户理解讲道内容的助手。用中文回答。"
    else:
        system_message = "You are a helpful assistant that answers questions about sermon content."
    
    # Prepare the prompt for GPT-4 based on language - enhanced to mention sermon title and date
    if language == "es":
        prompt = f"""
Responde a la siguiente pregunta basándote únicamente en los segmentos de sermón proporcionados. Si la respuesta no se encuentra en los segmentos, dilo claramente.

PREGUNTA DEL USUARIO: {query}

SEGMENTOS DEL SERMÓN:
{context}

Responde a la pregunta basándote únicamente en los segmentos de sermón proporcionados. Incluye referencias específicas al sermón (por título y fecha) y qué segmento(s) contienen la información (por ejemplo, "En el sermón 'El Amor de Dios' del 15 de enero de 2023, Segmento 3, el pastor explica..."). Mantén tu respuesta enfocada y concisa.
        """
    elif language == "zh":
        prompt = f"""
根据提供的讲道片段回答以下问题。如果在这些片段中找不到答案，请清楚地说明。

用户问题: {query}

讲道片段:
{context}

仅根据提供的讲道片段回答问题。包括具体引用哪个讲道（按标题和日期）和哪个片段包含信息（例如，"在2023年1月15日的'上帝的爱'讲道中，片段3中，牧师解释了..."）。保持回答重点明确和简洁。
        """
    else:
        prompt = f"""
Answer the following question based only on the provided sermon segments. If the answer cannot be found in the segments, say so clearly.

USER QUESTION: {query}

SERMON SEGMENTS:
{context}

Answer the question based only on the provided sermon segments. Include specific references to which sermon (by title and date) and segment(s) contain the information (e.g., "In the sermon 'God's Love' from January 15, 2023, Segment 3, the pastor explains..."). Keep your response focused and concise.
        """
    
    try:
        response = openai_client.chat.completions.create(
            model=COMPLETION_MODEL,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=700
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating AI answer: {str(e)}")

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

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Sermon Search API is running",
        "documentation": "/docs",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    # Check connections to external services
    try:
        # Check Pinecone - updated to use v6.0.2 API
        stats = pinecone_index.describe_index_stats()
        vector_count = stats.total_vector_count
        
        # Check OpenAI (minimal test)
        openai_client.embeddings.create(
            input="test",
            model=EMBEDDING_MODEL
        )
        
        # Check Bible references
        bible_refs = load_bible_references()
        bible_books_count = len(bible_refs)
        
        return {
            "status": "healthy",
            "pinecone": {
                "status": "connected",
                "index": PINECONE_INDEX_NAME,
                "vector_count": vector_count
            },
            "openai": {
                "status": "connected",
                "embedding_model": EMBEDDING_MODEL,
                "completion_model": COMPLETION_MODEL
            },
            "bible_references": {
                "status": "available" if bible_books_count > 0 else "not found",
                "books_count": bible_books_count
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# Modified to support title and date filtering
@app.get("/search", response_model=SearchResponse)
async def search(
    query: str = Query(..., description="The search query"),
    top_k: int = Query(SEARCH_TOP_K, description="Number of results to return"),
    min_score: float = Query(0.6, description="Minimum similarity score (0-1)"),
    title: Optional[str] = Query(None, description="Filter by sermon title (partial match)"),
    date: Optional[str] = Query(None, description="Filter by sermon date (YYYY-MM-DD)"),
    preacher: Optional[str] = Query(None, description="Filter by preacher name")
):
    """
    Search for sermon segments matching the query.
    Returns the most semantically similar segments from the sermon library.
    """
    start_time = time.time()
    
    try:
        # Generate embedding for the query
        query_embedding = generate_embedding(query)
        
        # Build filter dict based on parameters
        filter_dict = {}
        
        if title:
            # For Pinecone, we need to check if metadata contains this title
            # Note: This may need adjustment based on your Pinecone plan
            # Basic approach is exact match - more advanced would require
            # a different filtering mechanism that supports partial matches
            filter_dict["title"] = {"$eq": title}
        
        if date:
            filter_dict["publish_date"] = {"$eq": date}
            
        if preacher:
            filter_dict["preacher"] = {"$eq": preacher}
        
        # Search Pinecone - updated for v6.0.2 API
        search_response = pinecone_index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict if filter_dict else None
        )
        
        # Format results - updated for v6.0.2 API and enhanced with metadata
        results = []
        
        for match in search_response.matches:
            if match.score < min_score:
                continue
                
            metadata = match.metadata
            video_id = metadata.get("video_id", "")
            
            # Load additional metadata to get proper title and date
            enhanced_metadata = load_metadata(video_id)
            
            # Convert segment_ids to List[str] if needed
            segment_ids = metadata.get("segment_ids", [])
            if not isinstance(segment_ids, list):
                segment_ids = []
            
            results.append(SearchResult(
                video_id=video_id,
                title=enhanced_metadata.get("title", metadata.get("title", "Unknown Sermon")),
                url=get_youtube_timestamp_url(video_id, metadata.get("start_time", 0)),
                text=metadata.get("text", ""),
                start_time=metadata.get("start_time", 0),
                end_time=metadata.get("end_time", 0),
                similarity=match.score,
                chunk_index=metadata.get("chunk_index", 0),
                segment_ids=segment_ids,
                publish_date=enhanced_metadata.get("publish_date")
            ))
        
        return SearchResponse(
            query=query,
            results=results,
            total_results=len(results),
            processing_time=time.time() - start_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

# Modified to support title and date filtering
@app.post("/answer", response_model=AnswerResponse)
async def answer(request: AnswerRequest):
    """
    Generate an AI answer to a question based on sermon content.
    Searches for relevant sermon segments and uses them to create a response.
    """
    start_time = time.time()
    
    try:
        # Determine if we need to translate the query
        original_language = request.language
        needs_translation = original_language != "en"
        
        # Translate query to English if needed
        query = request.query
        if needs_translation:
            query = await translate_text(query, original_language, "en")
            print(f"Translated query from {original_language} to English: {query}")
        
        # Generate embedding for the translated query
        query_embedding = generate_embedding(query)
        
        # Build filter dict based on parameters
        filter_dict = {}
        
        if request.title:
            filter_dict["title"] = {"$eq": request.title}
        
        if request.date:
            filter_dict["publish_date"] = {"$eq": request.date}
            
        if request.preacher:
            filter_dict["preacher"] = {"$eq": request.preacher}
        
        # Search Pinecone - updated for v6.0.2 API
        search_response = pinecone_index.query(
            vector=query_embedding,
            top_k=request.top_k,
            include_metadata=True,
            filter=filter_dict if filter_dict else None
        )
        
        # Format search results - updated for v6.0.2 API and enhanced with metadata
        search_results = []
        
        for match in search_response.matches:
            if match.score < 0.5:  # Minimum threshold for relevance
                continue
                
            metadata = match.metadata
            video_id = metadata.get("video_id", "")
            
            # Load additional metadata to get proper title and date
            enhanced_metadata = load_metadata(video_id)
            
            # Convert segment_ids to List[str] if needed
            segment_ids = metadata.get("segment_ids", [])
            if not isinstance(segment_ids, list):
                segment_ids = []
            
            search_results.append(SearchResult(
                video_id=video_id,
                title=enhanced_metadata.get("title", metadata.get("title", "Unknown Sermon")),
                url=get_youtube_timestamp_url(video_id, metadata.get("start_time", 0)),
                text=metadata.get("text", ""),
                start_time=metadata.get("start_time", 0),
                end_time=metadata.get("end_time", 0),
                similarity=match.score,
                chunk_index=metadata.get("chunk_index", 0),
                segment_ids=segment_ids,
                publish_date=enhanced_metadata.get("publish_date")
            ))
        
        # Generate AI answer - always generate in English first, then translate if needed
        default_no_results = "No relevant sermon content found to answer this question."
        
        if search_results:
            # Always generate in English first for consistency
            answer_text = generate_ai_answer(query, search_results, "en")
            
            # Then translate to the requested language if needed
            if needs_translation:
                answer_text = await translate_text(answer_text, "en", original_language)
                print(f"Translated answer from English to {original_language}")
        else:
            # Handle no results case with appropriate translation
            answer_text = default_no_results
            if needs_translation:
                answer_text = await translate_text(default_no_results, "en", original_language)
        
        return AnswerResponse(
            query=request.query,  # Return the original untranslated query
            answer=answer_text,
            sources=search_results if request.include_sources else [],
            processing_time=time.time() - start_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Answer generation error: {str(e)}")

@app.get("/transcript/{video_id}")
async def get_transcript(
    video_id: str,
    language: Optional[str] = Query("en", description="Language for the transcript")
):
    """
    Get a sermon transcript with optional translation.
    Returns the transcript segments with timestamps.
    """
    try:
        # First, try to use the SRT file directly if it exists
        srt_file = os.path.join(SUBTITLES_DIR, f"{video_id}.srt")
        
        # Load metadata for better title and date
        enhanced_metadata = load_metadata(video_id)
        
        if os.path.exists(srt_file):
            # Parse the SRT file
            segments = []
            with open(srt_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split by double newline to get individual subtitle entries
            entries = content.strip().split('\n\n')
            
            for entry in entries:
                lines = entry.strip().split('\n')
                if len(lines) < 3:
                    continue
                
                # Parse the timestamp line (format: 00:00:00,000 --> 00:00:00,000)
                timestamp_line = lines[1]
                time_parts = timestamp_line.split(' --> ')
                if len(time_parts) != 2:
                    continue
                
                start_time_str, end_time_str = time_parts
                
                # Convert timestamp to seconds
                def time_to_seconds(time_str):
                    h, m, s = time_str.replace(',', '.').split(':')
                    return float(h) * 3600 + float(m) * 60 + float(s)
                
                start_time = time_to_seconds(start_time_str)
                end_time = time_to_seconds(end_time_str)
                
                # Get the text (might be multiple lines)
                text = ' '.join(lines[2:])
                
                # If language is not English, translate the text
                if language != "en":
                    try:
                        text = await translate_text(text, "en", language)
                    except Exception as e:
                        print(f"Translation error for segment: {str(e)}")
                        # Continue with untranslated text
                
                segments.append({
                    "start_time": start_time,
                    "end_time": end_time,
                    "text": text
                })
            
            return {
                "video_id": video_id,
                "title": enhanced_metadata.get("title", f"Sermon {video_id}"),
                "publish_date": enhanced_metadata.get("publish_date"),
                "language": language,
                "segments": segments,
                "total_segments": len(segments),
                "transcript_source": "srt_file"
            }
        
        # If SRT file doesn't exist, fall back to Pinecone query
        query_embedding = generate_embedding("sermon transcript")
        
        # Create a filter to match video_id
        filter_dict = {"video_id": {"$eq": video_id}}
        
        # Increase top_k to ensure we get ALL segments
        results = pinecone_index.query(
            vector=query_embedding,
            top_k=500,  # Get much more results to ensure complete coverage
            include_metadata=True,
            filter=filter_dict
        )
        
        if not results.matches:
            raise HTTPException(status_code=404, detail=f"Transcript not found for video: {video_id}")
        
        # Convert matches to transcript segments
        raw_segments = []
        for match in results.matches:
            metadata = match.metadata
            if metadata.get("video_id") == video_id and metadata.get("text"):
                text = metadata.get("text", "")
                
                # If language is not English, translate the text
                if language != "en":
                    try:
                        text = await translate_text(text, "en", language)
                    except Exception as e:
                        print(f"Translation error for segment: {str(e)}")
                        # Continue with untranslated text
                
                raw_segments.append({
                    "start_time": metadata.get("start_time", 0),
                    "end_time": metadata.get("end_time", 0),
                    "text": text,
                    "chunk_index": metadata.get("chunk_index", 0)
                })
        
        # Sort segments by start time
        raw_segments.sort(key=lambda x: x["start_time"])
        
        # Process segments to remove duplicates and fill gaps
        processed_segments = []
        
        if raw_segments:
            # Start with the first segment
            current_segment = raw_segments[0].copy()
            
            for i in range(1, len(raw_segments)):
                next_segment = raw_segments[i]
                
                # Check if current segment overlaps with next segment
                if current_segment["end_time"] >= next_segment["start_time"]:
                    # Find the overlap
                    current_text = current_segment["text"]
                    next_text = next_segment["text"]
                    
                    # Try to find where the overlap begins
                    overlap_found = False
                    
                    # Check for substantial text overlap (at least 20 characters)
                    for overlap_size in range(20, min(len(current_text), len(next_text)) // 2):
                        if current_text[-overlap_size:] == next_text[:overlap_size]:
                            # We found an overlap
                            overlap_found = True
                            
                            # Merge the segments
                            current_segment["text"] = current_text + next_text[overlap_size:]
                            current_segment["end_time"] = next_segment["end_time"]
                            break
                    
                    # If no clear overlap found but timestamps indicate they should connect
                    if not overlap_found and (next_segment["start_time"] - current_segment["end_time"]) < 2:
                        # Just append with a space
                        current_segment["text"] = current_text + " " + next_text
                        current_segment["end_time"] = next_segment["end_time"]
                else:
                    # No overlap - check if there's a gap
                    time_gap = next_segment["start_time"] - current_segment["end_time"]
                    
                    if time_gap > 5:  # Significant gap (more than 5 seconds)
                        # Add an indicator of missing content
                        processed_segments.append(current_segment)
                        processed_segments.append({
                            "start_time": current_segment["end_time"],
                            "end_time": next_segment["start_time"],
                            "text": "[...]",  # Indicator of gap
                            "is_gap": True
                        })
                        current_segment = next_segment.copy()
                    else:
                        # Small gap, just add current and start a new one
                        processed_segments.append(current_segment)
                        current_segment = next_segment.copy()
            
            # Add the final segment
            processed_segments.append(current_segment)
        
        return {
            "video_id": video_id,
            "title": enhanced_metadata.get("title", f"Sermon {video_id}"),
            "publish_date": enhanced_metadata.get("publish_date"),
            "language": language,
            "segments": processed_segments,
            "total_segments": len(processed_segments),
            "transcript_source": "pinecone"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving transcript: {str(e)}")

@app.get("/sermons")
async def list_sermons(
    limit: int = Query(100, description="Maximum number of sermons to return"),
    offset: int = Query(0, description="Offset for pagination"),
    sort_by: str = Query("date", description="Sort by 'date' or 'title'"),
    order: str = Query("desc", description="Sort order: 'asc' or 'desc'")
):
    """
    List available sermons in the library.
    Returns metadata for available sermons.
    """
    try:
        # Get index stats to confirm vector count
        stats = pinecone_index.describe_index_stats()
        total_vectors = stats.total_vector_count
        
        # Collect all unique sermon IDs with a systematic approach
        all_sermons = {}
        batch_size = 10000  # Max vectors to process per batch
        
        # First, get all unique video_ids using metadata filtering
        # We'll use different random vectors to get diverse results
        for i in range(0, (total_vectors // batch_size) + 1):
            # Create a varied random vector for each batch to get different results
            random_vector = [(i + j) / 1000.0 for j in range(1536)]
            
            response = pinecone_index.query(
                vector=random_vector,
                top_k=batch_size,  # Get a large batch
                include_metadata=True
            )
            
            # Process each match to extract sermon metadata
            for match in response.matches:
                metadata = match.metadata
                video_id = metadata.get("video_id", "")
                
                if video_id and video_id not in all_sermons:
                    # Load enhanced metadata
                    enhanced_metadata = load_metadata(video_id)
                    
                    all_sermons[video_id] = {
                        "video_id": video_id,
                        "title": enhanced_metadata.get("title", metadata.get("title", f"Sermon {video_id}")),
                        "channel": metadata.get("channel", "Unknown"),
                        "publish_date": enhanced_metadata.get("publish_date", metadata.get("publish_date", "")),
                        "url": f"https://www.youtube.com/watch?v={video_id}"
                    }
        
        # If we still don't have all sermons, try using filters directly
        # This approach helps ensure we get all unique sermons
        if len(all_sermons) < 400:  # Assuming we should have at least 400 sermons based on your count of 429
            # Get namespaces if your index uses them
            try:
                # For each unique video_id we've found so far, query for more related vectors
                existing_ids = list(all_sermons.keys())
                for video_id in existing_ids:
                    # Use filter to get vectors with this video_id
                    filter_dict = {"video_id": {"$eq": video_id}}
                    
                    result = pinecone_index.query(
                        vector=[0.1] * 1536,  # Placeholder vector
                        top_k=5,  # Just need a few to get metadata
                        include_metadata=True,
                        filter=filter_dict
                    )
                    
                    # Get any additional metadata
                    for match in result.matches:
                        metadata = match.metadata
                        if metadata.get("video_id") == video_id:
                            # Load enhanced metadata
                            enhanced_metadata = load_metadata(video_id)
                            
                            # Update sermon info with any additional metadata
                            all_sermons[video_id].update({
                                "title": enhanced_metadata.get("title", metadata.get("title", all_sermons[video_id]["title"])),
                                "channel": metadata.get("channel", all_sermons[video_id]["channel"]),
                                "publish_date": enhanced_metadata.get("publish_date", metadata.get("publish_date", all_sermons[video_id]["publish_date"]))
                            })
            except Exception as filter_err:
                # If filtering fails, log the error but continue
                print(f"Filter query error: {str(filter_err)}")
        
        # Convert to list and apply sorting and pagination
        sermon_list = list(all_sermons.values())
        
        # Apply sorting
        reverse_order = order.lower() == "desc"
        if sort_by.lower() == "title":
            sermon_list.sort(key=lambda x: x.get("title", "").lower(), reverse=reverse_order)
        else:  # Default to date
            sermon_list.sort(key=lambda x: x.get("publish_date", ""), reverse=reverse_order)
        
        # Apply pagination
        paginated_sermons = sermon_list[offset:offset + limit]
        
        return {
            "sermons": paginated_sermons,
            "total": len(sermon_list),
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing sermons: {str(e)}")

@app.get("/sermons/{video_id}")
async def get_sermon(video_id: str):
    """
    Get information about a specific sermon.
    Returns metadata and available chunks for the sermon.
    """
    try:
        # Load enhanced metadata
        enhanced_metadata = load_metadata(video_id)
        
        # Query Pinecone for chunks with this video_id
        # Use a filter to get only chunks for this sermon
        query_embedding = generate_embedding("sermon about faith")  # Generic query
        
        # Create a filter to match video_id - updated for v6.0.2 API
        filter_dict = {"video_id": {"$eq": video_id}}
        
        results = pinecone_index.query(
            vector=query_embedding,
            top_k=100,  # Get many results to find chunks from this sermon
            include_metadata=True,
            filter=filter_dict
        )
        
        # Updated for v6.0.2 API
        if not results.matches:
            raise HTTPException(status_code=404, detail=f"Sermon not found: {video_id}")
        
        # Get sermon metadata from the first match - updated for v6.0.2 API
        first_match = results.matches[0].metadata
        sermon_info = {
            "video_id": video_id,
            "title": enhanced_metadata.get("title", first_match.get("title", f"Sermon {video_id}")),
            "channel": first_match.get("channel", "Unknown"),
            "publish_date": enhanced_metadata.get("publish_date", first_match.get("publish_date", "")),
            "url": f"https://www.youtube.com/watch?v={video_id}"
        }
        
        # Collect all chunks - updated for v6.0.2 API
        chunks = []
        for match in results.matches:
            metadata = match.metadata
            if metadata.get("video_id") == video_id:
                # Convert segment_ids to List[str] if needed
                segment_ids = metadata.get("segment_ids", [])
                if not isinstance(segment_ids, list):
                    segment_ids = []
                
                chunks.append({
                    "chunk_index": metadata.get("chunk_index", 0),
                    "text": metadata.get("text", ""),
                    "start_time": metadata.get("start_time", 0),
                    "end_time": metadata.get("end_time", 0),
                    "segment_ids": segment_ids
                })
        
        # Sort chunks by start time
        chunks.sort(key=lambda x: x["start_time"])
        
        return {
            "sermon": sermon_info,
            "chunks": chunks,
            "total_chunks": len(chunks)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving sermon: {str(e)}")

# New endpoints for sermon metadata filtering

@app.get("/sermons/by-title")
async def list_sermons_by_title():
    """List all available sermon titles."""
    try:
        # Get all unique sermon titles from Pinecone
        # We need to use a query that will return diverse results
        # so we can get as many unique video_ids as possible
        random_vector = [0.1] * 1536  # Placeholder vector
        
        response = pinecone_index.query(
            vector=random_vector,
            top_k=1000,  # Get a large batch
            include_metadata=True
        )
        
        # Extract unique titles
        titles = {}
        for match in response.matches:
            metadata = match.metadata
            video_id = metadata.get("video_id", "")
            
            if video_id and video_id not in titles:
                # Load enhanced metadata
                enhanced_metadata = load_metadata(video_id)
                
                title = enhanced_metadata.get("title", metadata.get("title", f"Sermon {video_id}"))
                titles[video_id] = {
                    "video_id": video_id, 
                    "title": title,
                    "publish_date": enhanced_metadata.get("publish_date", metadata.get("publish_date", "")),
                    "url": f"https://www.youtube.com/watch?v={video_id}"
                }
        
        # Convert to list and sort by title
        title_list = list(titles.values())
        title_list.sort(key=lambda x: x["title"])
        
        return {
            "titles": title_list,
            "total": len(title_list)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing sermon titles: {str(e)}")

@app.get("/sermons/by-date")
async def list_sermons_by_date():
    """List all available sermon dates."""
    try:
        # Get all unique sermon dates from Pinecone
        random_vector = [0.1] * 1536  # Placeholder vector
        
        response = pinecone_index.query(
            vector=random_vector,
            top_k=1000,  # Get a large batch
            include_metadata=True
        )
        
        # Extract unique dates and their associated sermons
        dates = {}
        for match in response.matches:
            metadata = match.metadata
            video_id = metadata.get("video_id", "")
            
            if video_id and video_id not in dates:
                # Load enhanced metadata
                enhanced_metadata = load_metadata(video_id)
                
                publish_date = enhanced_metadata.get("publish_date", metadata.get("publish_date", ""))
                if publish_date:
                    # Use video_id as key to avoid duplicates
                    dates[video_id] = {
                        "video_id": video_id,
                        "date": publish_date,
                        "title": enhanced_metadata.get("title", metadata.get("title", f"Sermon {video_id}")),
                        "url": f"https://www.youtube.com/watch?v={video_id}"
                    }
        
        # Convert to list and sort by date
        date_list = list(dates.values())
        date_list.sort(key=lambda x: x["date"], reverse=True)
        
        # Group by year and month for easier UI presentation
        grouped_dates = {}
        for sermon in date_list:
            date_str = sermon["date"]
            try:
                # Try to extract year and month from date string
                if "-" in date_str:
                    year, month, _ = date_str.split("-", 2)
                elif "/" in date_str:
                    month, _, year = date_str.split("/", 2)
                else:
                    # If unrecognized format, use "Unknown" as year/month
                    year, month = "Unknown", "Unknown"
                
                # Create year group if it doesn't exist
                if year not in grouped_dates:
                    grouped_dates[year] = {}
                
                # Create month group if it doesn't exist
                if month not in grouped_dates[year]:
                    grouped_dates[year][month] = []
                
                # Add sermon to the appropriate group
                grouped_dates[year][month].append(sermon)
            except Exception:
                # If date parsing fails, add to Unknown category
                if "Unknown" not in grouped_dates:
                    grouped_dates["Unknown"] = {"Unknown": []}
                grouped_dates["Unknown"]["Unknown"].append(sermon)
        
        return {
            "dates": date_list,
            "grouped_dates": grouped_dates,
            "total": len(date_list)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing sermon dates: {str(e)}")

# Bible reference endpoints
@app.get("/bible/stats", response_model=BibleReferenceStats)
async def get_bible_reference_stats():
    """Get statistics about Bible references in sermons."""
    try:
        stats = get_bible_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating Bible stats: {str(e)}")

@app.get("/bible/books")
async def list_bible_books():
    """List all Bible books with reference counts."""
    try:
        all_references = load_bible_references()
        books = [
            {
                "book": book,
                "count": len(references)
            }
            for book, references in all_references.items()
        ]
        
        # Sort books by count, descending
        books.sort(key=lambda x: x["count"], reverse=True)
        
        return {
            "books": books,
            "total_books": len(books),
            "total_references": sum(book["count"] for book in books)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing Bible books: {str(e)}")

@app.get("/bible/books/{book}")
async def get_book_references(book: str):
    """Get all references for a specific Bible book."""
    try:
        all_references = load_bible_references()
        
        if book not in all_references:
            raise HTTPException(status_code=404, detail=f"Book not found: {book}")
        
        # Get references and enhance with metadata
        references = all_references[book]
        enhanced_references = []
        
        for ref in references:
            video_id = ref.get("video_id", "")
            
            # Load sermon metadata for each reference
            metadata = load_metadata(video_id)
            
            enhanced_references.append({
                **ref,
                "sermon_title": metadata.get("title", f"Sermon {video_id}"),
                "url": get_youtube_timestamp_url(video_id, ref.get("start_time", 0))
            })
        
        # Group references by chapter
        chapters = defaultdict(list)
        for ref in enhanced_references:
            chapter_key = str(ref.get("chapter", "unknown"))
            chapters[chapter_key].append(ref)
        
        # Sort verses within each chapter - handle different verse formats
        for chapter_key, chapter_refs in chapters.items():
            # Define a custom sorting function that handles different verse formats
            def verse_sort_key(ref):
                verse = ref.get("verse", 0)
                if verse is None:
                    return 0
                    
                # Convert to string if it's not already
                verse_str = str(verse)
                
                # Get the first number in case of ranges
                if '-' in verse_str:
                    return int(verse_str.split('-')[0])
                else:
                    return int(verse_str or 0)
            
            # Sort using our custom function
            chapter_refs.sort(key=verse_sort_key)
        
        return {
            "book": book,
            "total_references": len(enhanced_references),
            "chapters": dict(chapters),
            "references": enhanced_references
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving book references: {str(e)}")

@app.get("/bible/references/{reference_id}")
async def get_reference(reference_id: str):
    """Get all occurrences of a specific Bible reference (like "John 3:16" or "Romans 8")."""
    try:
        book = None
        chapter = None
        verse = None
        
        # Parse reference ID format: "Book_Chapter_Verse" or "Book_Chapter" or just "Book"
        parts = reference_id.split('_')
        if len(parts) >= 1:
            book = parts[0]
        if len(parts) >= 2:
            try:
                chapter = int(parts[1])
            except ValueError:
                chapter = None
        if len(parts) >= 3:
            try:
                verse = int(parts[2])
            except ValueError:
                verse = None
        
        if not book:
            raise HTTPException(status_code=400, detail="Invalid reference format")
        
        # Load references for the book
        all_references = load_bible_references()
        if book not in all_references:
            raise HTTPException(status_code=404, detail=f"Book not found: {book}")
        
        book_references = all_references[book]
        
        # Filter by chapter and verse if provided
        filtered_references = book_references
        if chapter is not None:
            filtered_references = [ref for ref in filtered_references if ref.get("chapter") == chapter]
        if verse is not None:
            filtered_references = [ref for ref in filtered_references if ref.get("verse") == verse]
        
        if not filtered_references:
            raise HTTPException(status_code=404, detail=f"No references found for {reference_id}")
        
        # Enhance references with metadata
        enhanced_references = []
        for ref in filtered_references:
            video_id = ref.get("video_id", "")
            
            # Load sermon metadata
            metadata = load_metadata(video_id)
            
            enhanced_references.append({
                **ref,
                "sermon_title": metadata.get("title", f"Sermon {video_id}"),
                "sermon_date": metadata.get("publish_date", None),
                "url": get_youtube_timestamp_url(video_id, ref.get("start_time", 0))
            })
        
        # Group by sermon (video_id)
        sermons = defaultdict(list)
        for ref in enhanced_references:
            sermons[ref["video_id"]].append(ref)
        
        # Find related references (same chapter, different verses)
        related_references = []
        if chapter is not None:
            # Get other verses from the same chapter
            related = [
                ref for ref in book_references 
                if ref.get("chapter") == chapter and (verse is None or ref.get("verse") != verse)
            ]
            
            # Count occurrences by verse
            verse_counts = Counter(ref.get("verse") for ref in related if ref.get("verse") is not None)
            
            # Format related references
            for verse_num, count in verse_counts.most_common(10):  # Get top 10 related verses
                related_references.append({
                    "book": book,
                    "chapter": chapter,
                    "verse": verse_num,
                    "reference_text": f"{book} {chapter}:{verse_num}",
                    "count": count
                })
        
        # Format reference display text
        display_text = book.replace("_", " ")
        if chapter is not None:
            display_text += f" {chapter}"
            if verse is not None:
                display_text += f":{verse}"
        
        return {
            "reference_id": reference_id,
            "display_text": display_text,
            "book": book,
            "chapter": chapter,
            "verse": verse,
            "total_occurrences": len(enhanced_references),
            "occurrences_by_sermon": dict(sermons),
            "all_occurrences": enhanced_references,
            "related_references": related_references
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving reference: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)