import os
import time
import glob
import json
import re
from typing import List, Dict, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import time
from datetime import datetime, timedelta
import calendar

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
    version="1.1.0"  # Updated version number to reflect enhancements
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
    suggested_queries: List[str] = []  # Added field for suggested queries
    sermon_date: Optional[str] = None  # Added field for identified date in human-readable format
    sermon_title: Optional[str] = None  # Added field for identified title

class AnswerRequest(BaseModel):
    query: str = Field(..., description="The question to answer based on sermon content")
    top_k: int = Field(5, description="Number of search results to consider")
    include_sources: bool = Field(True, description="Whether to include source information in the response")
    language: str = Field("en", description="Language for the response (en, es, zh)")

class AnswerResponse(BaseModel):
    query: str
    answer: str
    sources: List[SearchResult] = []
    processing_time: float
    suggested_queries: List[str] = []  # Added field for suggested queries when no direct answer is found

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

# New functions for query processing
def preprocess_query(query: str) -> Tuple[str, Optional[int], Optional[str]]:
    """
    Preprocesses the search query to detect date references and title references.
    
    Args:
        query: The original user query
        
    Returns:
        Tuple containing:
        - processed_query: The cleaned query for embedding
        - date_filter: Unix timestamp for date filtering (if applicable)
        - title_filter: Title string for filtering (if applicable)
    """
    # Initialize return values
    processed_query = query
    date_filter = None
    title_filter = None
    
    # Check for date references first
    date_filter, human_readable_date = extract_date_reference(query)
    
    # Check for specific title patterns first
    title_match = re.search(r'(?:sermon|message|talk)(?:\s+(?:about|on|titled|called|named))?\s+["\']?([^"\'?.]+)["\']?', query, re.IGNORECASE)
    if title_match:
        title_filter = title_match.group(1).strip()
        # Remove the title reference from the query
        processed_query = re.sub(r'(?:sermon|message|talk)(?:\s+(?:about|on|titled|called|named))?\s+["\']?([^"\'?.]+)["\']?', '', processed_query, flags=re.IGNORECASE).strip()
    else:
        # Check if the query itself might be a sermon title or part of a sermon title
        # This allows direct searches like "The Power of Prayer" without needing "sermon titled"
        metadata_files = glob.glob(os.path.join(METADATA_DIR, "*_metadata.json"))
        for metadata_file in metadata_files:
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    sermon_title = metadata.get("title", "").lower()
                    query_lower = query.lower()
                    
                    # Check if the query is contained in or similar to the sermon title
                    if query_lower in sermon_title or sermon_title in query_lower:
                        words_in_query = set(query_lower.split())
                        words_in_title = set(sermon_title.split())
                        # If there's a significant word overlap, consider it a match
                        if len(words_in_query & words_in_title) >= min(2, len(words_in_query)):
                            title_filter = query
                            break
            except Exception as e:
                print(f"Error reading metadata for title matching: {str(e)}")
    
    return processed_query, date_filter, title_filter


def extract_date_reference(query: str) -> tuple[Optional[int], Optional[str]]:
    """
    Extract date references from the query and convert to Unix timestamp.
    Also returns a human-readable date string.
    
    Args:
        query: The user query
        
    Returns:
        Tuple containing:
        - Unix timestamp if a date reference is found, None otherwise
        - Human-readable date string if date reference is found, None otherwise
    """
    today = datetime.now()
    human_readable_date = None
    date_filter = None
    
    # Check for "last Sunday", "this Sunday", etc.
    day_match = re.search(r'(?:last|this|previous|past|next)\s+(Sunday|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday)', query, re.IGNORECASE)
    if day_match:
        day_name = day_match.group(1).capitalize()
        day_num = list(calendar.day_name).index(day_name)
        
        # Calculate the date for the referenced day
        if "last" in day_match.group(0).lower() or "previous" in day_match.group(0).lower() or "past" in day_match.group(0).lower():
            # Last week's day
            days_diff = (today.weekday() + 1) % 7 + (7 - day_num) % 7
            if days_diff == 0:
                days_diff = 7  # If today is the same day, go back a week
            target_date = today - timedelta(days=days_diff)
        elif "next" in day_match.group(0).lower():
            # Next week's day
            days_diff = (day_num - today.weekday() - 1) % 7
            if days_diff == 0:
                days_diff = 7  # If today is the same day, go forward a week
            target_date = today + timedelta(days=days_diff)
        else:
            # This week's day
            days_diff = (day_num - today.weekday()) % 7
            target_date = today + timedelta(days=days_diff)
            if days_diff > 0:
                # If the day hasn't occurred yet this week, go back to last week
                target_date -= timedelta(days=7)
        
        # Convert to Unix timestamp (seconds since epoch)
        date_filter = int(target_date.timestamp())
        human_readable_date = target_date.strftime('%B %d, %Y')
        
        return date_filter, human_readable_date
    
    # Check for specific dates like "May 11th, 2025" or "2025-05-11"
    # Format: Month Day, Year
    date_match = re.search(r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})', query, re.IGNORECASE)
    if date_match:
        month_name = date_match.group(1).capitalize()
        month_num = list(calendar.month_name).index(month_name)
        day = int(date_match.group(2))
        year = int(date_match.group(3))
        
        try:
            target_date = datetime(year, month_num, day)
            date_filter = int(target_date.timestamp())
            human_readable_date = target_date.strftime('%B %d, %Y')
            return date_filter, human_readable_date
        except ValueError:
            # Invalid date, e.g., February 30
            return None, None
    
    # Format: YYYY-MM-DD
    iso_match = re.search(r'(\d{4})-(\d{1,2})-(\d{1,2})', query)
    if iso_match:
        year = int(iso_match.group(1))
        month = int(iso_match.group(2))
        day = int(iso_match.group(3))
        
        try:
            target_date = datetime(year, month, day)
            date_filter = int(target_date.timestamp())
            human_readable_date = target_date.strftime('%B %d, %Y')
            return date_filter, human_readable_date
        except ValueError:
            return None, None
    
    # Format: MM/DD/YYYY
    us_match = re.search(r'(\d{1,2})/(\d{1,2})/(\d{4})', query)
    if us_match:
        month = int(us_match.group(1))
        day = int(us_match.group(2))
        year = int(us_match.group(3))
        
        try:
            target_date = datetime(year, month, day)
            date_filter = int(target_date.timestamp())
            human_readable_date = target_date.strftime('%B %d, %Y')
            return date_filter, human_readable_date
        except ValueError:
            return None, None
    
    # Handle relative dates like "yesterday", "today", "last week"
    if re.search(r'\byesterday\b', query, re.IGNORECASE):
        target_date = today - timedelta(days=1)
        date_filter = int(target_date.timestamp())
        human_readable_date = target_date.strftime('%B %d, %Y')
        return date_filter, human_readable_date
    
    if re.search(r'\btoday\b', query, re.IGNORECASE):
        date_filter = int(today.timestamp())
        human_readable_date = today.strftime('%B %d, %Y')
        return date_filter, human_readable_date
    
    if re.search(r'\blast\s+week\b', query, re.IGNORECASE):
        target_date = today - timedelta(days=7)
        date_filter = int(target_date.timestamp())
        human_readable_date = target_date.strftime('%B %d, %Y')
        return date_filter, human_readable_date
    
    # No date reference found
    return None, None

def format_date_for_human(timestamp: Optional[int]) -> Optional[str]:
    """Convert a Unix timestamp to a human-readable date string."""
    if timestamp is None:
        return None
    
    try:
        return datetime.fromtimestamp(timestamp).strftime('%B %d, %Y')
    except (ValueError, TypeError, OverflowError):
        # Handle invalid timestamps
        return None

def get_proper_date_from_metadata(metadata: Dict[str, Any]) -> Optional[int]:
    """Extract and validate a date from metadata."""
    publish_date = metadata.get("publish_date")
    
    # Check if publish_date is a reasonable Unix timestamp
    # Unix timestamps should be between 1970 and current time plus some margin for future dates
    current_time = int(time.time())
    min_valid_timestamp = 0  # Jan 1, 1970
    max_valid_timestamp = current_time + 31536000  # Current time + 1 year (for future scheduled sermons)
    
    if publish_date is not None:
        try:
            publish_date = int(publish_date)
            if min_valid_timestamp <= publish_date <= max_valid_timestamp:
                return publish_date
        except (ValueError, TypeError):
            pass
    
    # If we couldn't get a valid date, check if there's a date in the title
    title = metadata.get("title", "")
    
    # Look for dates in titles like "Sermon - January 15, 2025"
    date_match = re.search(r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})', title, re.IGNORECASE)
    if date_match:
        month_name = date_match.group(1).capitalize()
        month_num = list(calendar.month_name).index(month_name)
        day = int(date_match.group(2))
        year = int(date_match.group(3))
        
        try:
            target_date = datetime(year, month_num, day)
            return int(target_date.timestamp())
        except ValueError:
            pass
    
    # Look for ISO-style dates in titles (YYYY-MM-DD)
    iso_match = re.search(r'(\d{4})-(\d{1,2})-(\d{1,2})', title)
    if iso_match:
        year = int(iso_match.group(1))
        month = int(iso_match.group(2))
        day = int(iso_match.group(3))
        
        try:
            target_date = datetime(year, month, day)
            return int(target_date.timestamp())
        except ValueError:
            pass
    
    # No valid date found
    return None

def format_response_with_suggestions(answer_text: str, suggested_queries: List[str]) -> str:
    """
    Formats the response text to include suggested queries in a user-friendly way.
    
    Args:
        answer_text: The original answer text
        suggested_queries: List of suggested queries
        
    Returns:
        Formatted answer text with suggestions
    """
    if not suggested_queries:
        return answer_text
    
    suggestions_text = "\n\n**Bible-Based Questions You Might Find Helpful:**\n"
    for i, query in enumerate(suggested_queries, 1):
        suggestions_text += f"\n{i}. {query}"
    
    return answer_text + suggestions_text

def generate_ai_answer_with_suggestions(query: str, search_results: List[SearchResult], suggested_queries: List[str] = [], language: str = "en") -> str:
    """
    Generate an AI answer that includes suggested queries if needed.
    
    Args:
        query: The user's query
        search_results: List of search results
        suggested_queries: List of suggested alternative queries
        language: Target language code
        
    Returns:
        Generated answer text with suggestions if applicable
    """
    # If we have search results, generate a normal answer
    if search_results:
        # Generate the main answer using enhanced_ai_answer
        answer_text = generate_enhanced_ai_answer(query, search_results, language)
        
        # If we also have suggested queries (related queries), add them to the end
        if suggested_queries:
            answer_text = format_response_with_suggestions(answer_text, suggested_queries)
        
        return answer_text
    
    # If no search results, generate a no-results message with suggestions
    no_results_message = generate_no_results_message(query, suggested_queries, language)
    return format_response_with_suggestions(no_results_message, suggested_queries)

def generate_suggested_queries_with_results(original_query: str, max_suggestions: int = 3) -> Tuple[List[str], List[SearchResult]]:
    """
    Generate suggested search queries and also return sample results for those queries.
    
    Args:
        original_query: The user's original query that returned no results
        max_suggestions: Maximum number of suggestions to return
        
    Returns:
        Tuple of (suggested queries list, sample results list)
    """
    # Generate basic suggested queries
    suggestions = generate_suggested_queries(original_query, max_suggestions)
    
    # For each suggestion, try to get at least one result
    all_results = []
    valid_suggestions = []
    
    for suggestion in suggestions:
        try:
            # Get embedding for the suggested query
            query_embedding = generate_embedding(suggestion)
            
            # Search Pinecone with a lower threshold
            search_response = pinecone_index.query(
                vector=query_embedding,
                top_k=1,  # Just get one result to check if there are any
                include_metadata=True
            )
            
            # If we got any results, keep this suggestion
            if search_response.matches and len(search_response.matches) > 0:
                match = search_response.matches[0]
                metadata = match.metadata
                video_id = metadata.get("video_id", "")
                
                # Load additional metadata
                enhanced_metadata = load_metadata(video_id)
                
                # Convert segment_ids to List[str] if needed
                segment_ids = metadata.get("segment_ids", [])
                if not isinstance(segment_ids, list):
                    segment_ids = []
                
                result = SearchResult(
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
                )
                
                valid_suggestions.append(suggestion)
                all_results.append(result)
        except Exception as e:
            print(f"Error testing suggestion '{suggestion}': {str(e)}")
    
    # If we couldn't get any valid suggestions, fall back to the original suggestions
    if not valid_suggestions:
        return suggestions, []
    
    return valid_suggestions, all_results

def generate_suggested_queries(original_query: str, max_suggestions: int = 3) -> List[str]:
    """
    Generate suggested search queries that are specifically focused on Biblical content
    when the original query returns no results.
    
    Args:
        original_query: The user's original query that returned no results
        max_suggestions: Maximum number of suggestions to return
        
    Returns:
        List of suggested queries
    """
    try:
        # Use GPT-4o to generate Bible-passage-focused suggestions
        response = openai_client.chat.completions.create(
            model=COMPLETION_MODEL,
            messages=[
                {"role": "system", "content": """You are a helpful assistant for an Independent Fundamental Baptist (IFB) 
                church sermon search engine. A user's search query returned no results. Generate 3 alternative search 
                queries related to their original query that are more likely to match sermon content.
                
                Follow these guidelines:
                1. Always try to connect the query to a relevant Bible passage or principle
                2. Focus on core Biblical topics, spiritual concepts, and common sermon themes in IFB churches
                3. Include at least one suggestion that references a specific Bible book or passage
                4. Keep suggestions concise and directly related to the original intent
                5. Return only the queries without explanations, one per line
                
                Example 1:
                Original query: "dealing with anxiety"
                Suggestions:
                What does Philippians 4:6-7 teach about worry?
                Biblical wisdom on overcoming fear
                Finding peace in God's promises
                
                Example 2:
                Original query: "salvation requirements"
                Suggestions: 
                What does the Bible say about salvation through faith?
                Romans 10:9-10 and the path to salvation
                How to be saved according to scripture"""},
                {"role": "user", "content": f"Original query: '{original_query}'\nPlease suggest 3 alternative search queries."}
            ],
            temperature=0.7,
            max_tokens=150
        )
        
        # Parse the response to extract the suggested queries
        suggestion_text = response.choices[0].message.content.strip()
        suggestions = [q.strip() for q in suggestion_text.split('\n') if q.strip()]
        
        # Limit to the requested number of suggestions
        return suggestions[:max_suggestions]
        
    except Exception as e:
        print(f"Error generating suggestions: {str(e)}")
        # Fallback suggestions if API call fails
        return [
            "What does the Bible teach about faith?",
            "Finding God's will in scripture",
            "Biblical principles for Christian living"
        ]

def is_off_topic_query(query: str) -> bool:
    """
    Determine if a query is completely off-topic for sermon content.
    
    Args:
        query: The search query
        
    Returns:
        True if the query is deemed off-topic, False otherwise
    """
    # Simple keyword-based approach - could be enhanced with embeddings comparison
    off_topic_indicators = [
        "recipe", "food", "game", "sports", "movie", "film", "tv show", 
        "television", "stock market", "investment", "car", "vehicle",
        "computer", "technology", "politics", "election", "vacation", 
        "travel", "hotel", "restaurant", "shopping", "buy", "sell"
    ]
    
    # Check if query contains off-topic indicators
    query_lower = query.lower()
    for indicator in off_topic_indicators:
        if indicator in query_lower:
            return True
    
    return False

def generate_no_results_message(query: str, suggested_queries: List[str], language: str = "en") -> str:
    """
    Generate a helpful message when no results are found for a query.
    
    Args:
        query: The original query
        suggested_queries: List of suggested alternative queries
        language: Target language code
        
    Returns:
        A helpful message with suggestions
    """
    # Check if query is off-topic
    if is_off_topic_query(query):
        if language == "es":
            return """Lo siento, no pude encontrar contenido de sermones relacionado con tu pregunta. 
            Nuestro sistema está diseñado para responder preguntas sobre temas bíblicos y espirituales 
            discutidos en sermones. Aquí hay algunas preguntas que podrías probar:"""
        elif language == "zh":
            return """很抱歉，我找不到与您的问题相关的讲道内容。我们的系统旨在回答有关在讲道中讨论的圣经和属灵主题的问题。
            以下是一些您可以尝试的问题："""
        else:
            return """I'm sorry, I couldn't find sermon content related to your question. 
            Our system is designed to answer questions about biblical and spiritual topics 
            discussed in sermons. Here are some questions you might try:"""
    else:
        if language == "es":
            return f"""No encontré contenido de sermones que responda directamente a tu pregunta sobre "{query}". 
            Aquí hay algunas preguntas relacionadas que podrían darte información relevante:"""
        elif language == "zh":
            return f"""我没有找到直接回答您关于"{query}"问题的讲道内容。以下是一些可能为您提供相关信息的相关问题："""
        else:
            return f"""I didn't find sermon content that directly answers your question about "{query}". 
            Here are some related questions that might give you relevant information:"""

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

def generate_enhanced_ai_answer(query: str, search_results: List[SearchResult], language: str = "en") -> str:
    """
    Generate an enhanced AI answer based on the search results in the specified language.
    
    Args:
        query: The user's query
        search_results: List of search results
        language: Target language code (en, es, zh)
        
    Returns:
        Generated answer text
    """
    # Group results by sermon to provide better context
    sermons = {}
    for result in search_results:
        if result.video_id not in sermons:
            sermons[result.video_id] = {
                "title": result.title,
                "date": result.publish_date,
                "segments": []
            }
        sermons[result.video_id]["segments"].append(result)
    
    # Sort segments within each sermon by start_time
    for sermon_id, sermon_data in sermons.items():
        sermon_data["segments"].sort(key=lambda x: x.start_time)
    
    # Prepare the context from search results with sermon grouping
    context_parts = []
    for sermon_id, sermon_data in sermons.items():
        # Add sermon header with properly formatted date
        date_str = ""
        if sermon_data["date"]:
            try:
                date_obj = datetime.fromtimestamp(sermon_data["date"])
                # Only add the date if it's a reasonable year (to avoid 1970 issues)
                if 1990 <= date_obj.year <= 2030:
                    date_str = f" ({date_obj.strftime('%B %d, %Y')})"
            except (ValueError, TypeError, OverflowError):
                # If date conversion fails, don't include it
                pass
                
        context_parts.append(f"SERMON: {sermon_data['title']}{date_str}")
        
        # Add segments
        for i, segment in enumerate(sermon_data["segments"]):
            time_str = f"{format_time(segment.start_time)} - {format_time(segment.end_time)}"
            context_parts.append(f"  Segment {i+1} ({time_str}):\n  {segment.text}")
        
        # Add separator between sermons
        context_parts.append("---")
    
    # Join all context parts
    context = "\n\n".join(context_parts)
    
    # Set system message based on language
    if language == "es":
        system_message = """Eres un asistente experto en contenido de sermones para una iglesia Bautista Fundamental Independiente. 
        Tu tarea es proporcionar respuestas detalladas y matizadas basadas únicamente en los segmentos de sermón proporcionados. 
        Si la información no está presente en los segmentos, debes indicarlo claramente. Proporciona citas contextuales 
        de los sermones para respaldar tus puntos.
        
        Cuando respondas, sigue estas pautas:
        1. Utiliza únicamente la información explícitamente indicada en los segmentos del sermón.
        2. Cita partes específicas de los sermones usando "comillas" para respaldar puntos clave.
        3. Cuando te refieras al que predica, usa términos como "el predicador", "el pastor", "el misionero", o "el evangelista" 
           según corresponda al contexto, en lugar de términos genéricos como "el orador". Si el sermón menciona específicamente quién
           está predicando, usa ese título y nombre.
        4. Indica qué sermón contiene la información (por ejemplo, "En el sermón titulado 'Fe en Acción'...").
        
        IMPORTANTE: Cuando menciones fechas de sermones, solo menciona fechas si están claramente especificadas en los segmentos 
        proporcionados. Si no estás seguro de una fecha, omítela completamente. Nunca inventes o asumas fechas."""
    elif language == "zh":
        system_message = """你是一位专门研究独立基要浸信会讲道内容的专家助手。你的任务是仅根据提供的讲道片段提供详细和有深度的答案。
        如果信息不在片段中，你应该清楚地说明。提供讲道中的上下文引用来支持你的观点。
        
        回答时，请遵循以下指导原则：
        1. 仅使用讲道片段中明确表述的信息。
        2. 用"引号"引用讲道中的特定部分来支持关键观点。
        3. 在提及讲道者时，请使用"牧师"、"传道人"、"宣教士"或"布道家"等术语，而不是泛化的"讲者"。
           如果讲道中明确提到讲道者的身份，请使用相应的头衔和名称。
        4. 提及包含该信息的讲道（例如，"在题为'信心与行动'的讲道中..."）。
        
        重要：当提及讲道日期时，只有在片段中明确指定的情况下才能提及日期。如果你不确定日期，请完全省略它。永远不要发明或假设日期。"""
    else:
        system_message = """You are an expert sermon content assistant for an Independent Fundamental Baptist church. 
        Your task is to provide detailed and nuanced answers based solely on the provided sermon segments.
        If information is not present in the segments, you must clearly indicate this. 
        Provide contextual quotes from the sermons to support your points.
        
        When answering, follow these guidelines:
        1. Only use information explicitly stated in the sermon segments.
        2. Quote specific parts of the sermons using "quotation marks" to support key points.
        3. When referring to who is preaching, use terms like "the preacher," "the pastor," "the missionary," or "the evangelist" 
           as appropriate to the context, rather than the generic "the speaker." If the sermon specifically mentions who is preaching, 
           use that title and name. Most sermons are from Pastor Mann, but others may be from missionaries, evangelists, or lay preachers.
        4. Reference which sermon contains the information (e.g., "In the sermon titled 'Faith in Action'...").
        5. If the question asks about a specific sermon by date or title, prioritize content from that sermon.
        6. If answering requires theological interpretation beyond what's in the segments, clearly indicate this.
        7. Keep your answer focused and organized, with clear structure.
        8. For scripture references, provide the book, chapter, and verse as mentioned in the sermon.
        
        IMPORTANT: When mentioning sermon dates, only mention dates if they are clearly specified in the provided segments.
        If you are unsure about a date, omit it completely. Never invent or assume dates. If a sermon's date is not
        clearly provided or seems incorrect (like very old dates such as 1970), do not mention the date at all."""
    
    # Prepare the prompt for GPT-4o
    prompt = f"""
Answer the following question based only on the provided sermon segments. 

USER QUESTION: {query}

SERMON CONTENT:
{context}

Answer the question based only on the provided sermon content. Include specific references to which sermon(s) contain the information. Keep your response focused and well-organized.

IMPORTANT REMINDERS: 
1. Only mention dates if they appear in the sermon title or content.
2. When referring to who is preaching, use appropriate terms like "the preacher," "the pastor," "the missionary," or "the evangelist" rather than "the speaker." If the sermon mentions a specific name or title, use that.
3. Most sermons come from Pastor Mann, but some are from missionaries, evangelists, or lay preachers. Only attribute to a specific person if you can clearly determine who is speaking from the sermon content.
"""
    
    try:
        response = openai_client.chat.completions.create(
            model=COMPLETION_MODEL,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1000
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
        "version": "1.1.0"  # Updated version number
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

@app.get("/search", response_model=SearchResponse)
async def search(
    query: str = Query(..., description="The search query"),
    top_k: int = Query(SEARCH_TOP_K, description="Number of results to return"),
    min_score: float = Query(0.6, description="Minimum similarity score (0-1)")
):
    """
    Search for sermon segments matching the query.
    Returns the most semantically similar segments from the sermon library.
    Now enhanced with date and title awareness, and suggested queries when no results found.
    """
    start_time = time.time()
    
    try:
        # Pre-process query for date/title recognition
        processed_query = query
        title_filter = None
        
        # Extract date information with improved function
        date_filter, human_readable_date = extract_date_reference(query)
        
        # Check for title references
        title_match = re.search(r'(?:sermon|message|talk)(?:\s+(?:about|on|titled|called|named))?\s+["\']?([^"\'?.]+)["\']?', query, re.IGNORECASE)
        if title_match:
            title_filter = title_match.group(1).strip()
            # Remove the title reference from the query
            processed_query = re.sub(r'(?:sermon|message|talk)(?:\s+(?:about|on|titled|called|named))?\s+["\']?([^"\'?.]+)["\']?', '', processed_query, flags=re.IGNORECASE).strip()
        
        # Generate embedding for the processed query
        query_embedding = generate_embedding(processed_query)
        
        # Build Pinecone filter based on detected date
        pinecone_filter = {}
        if date_filter:
            # Use approximate date matching (within 48 hours)
            day_start = date_filter - (48 * 60 * 60)  # Subtract two days in seconds
            day_end = date_filter + (48 * 60 * 60)    # Add two days in seconds
            pinecone_filter["publish_date"] = {"$gte": day_start, "$lte": day_end}
        
        # Search Pinecone
        search_response = pinecone_index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=pinecone_filter if pinecone_filter else None
        )
        
        # Format results
        results = []
        
        for match in search_response.matches:
            if match.score < min_score:
                continue
                
            metadata = match.metadata
            video_id = metadata.get("video_id", "")
            
            # Load additional metadata to get proper title and date
            enhanced_metadata = load_metadata(video_id)
            
            # Apply title filtering if specified
            if title_filter:
                # Use partial matching instead of exact matching
                title_lower = enhanced_metadata.get("title", "").lower()
                title_filter_lower = title_filter.lower()
                
                # Check if title_filter is a substantial part of the title or vice versa
                if title_filter_lower not in title_lower and title_lower not in title_filter_lower:
                    # Check word-by-word match (for cases like "faith sermon" matching "A Sermon About Faith")
                    title_words = set(title_lower.split())
                    filter_words = set(title_filter_lower.split())
                    significant_words = filter_words - {'sermon', 'message', 'about', 'on', 'the', 'a', 'an', 'and', 'or', 'of', 'in', 'for', 'with'}
                    
                    # If no significant match in important words, skip this result
                    if len(title_words.intersection(significant_words)) < min(1, len(significant_words)):
                        continue
            # Validate and get proper date
            publish_date = get_proper_date_from_metadata(enhanced_metadata)
            
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
                publish_date=publish_date
            ))
        
        # Handle no results case by generating suggested queries with samples
        suggested_queries = []
        if len(results) == 0:
            suggested_queries, sample_results = generate_suggested_queries_with_results(query)
            # If we have sample results, include the first one to give the user something useful
            if sample_results:
                results.append(sample_results[0])
        
        return SearchResponse(
            query=query,
            results=results,
            total_results=len(results),
            processing_time=time.time() - start_time,
            suggested_queries=suggested_queries,
            sermon_date=human_readable_date,
            sermon_title=title_filter
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

@app.post("/answer", response_model=AnswerResponse)
async def answer(request: AnswerRequest):
    """
    Generate an AI answer to a question based on sermon content.
    Searches for relevant sermon segments and uses them to create a response.
    Enhanced with date and title awareness, better context preparation, and suggested queries directly in the response.
    """
    start_time = time.time()
    
    try:
        # Determine if we need to translate the query
        original_language = request.language
        needs_translation = original_language != "en"
        
        # Process query to detect date/title references
        query = request.query
        processed_query = query
        
        # Extract date and title information
        date_filter, human_readable_date = extract_date_reference(query)
        
        # Check for title references
        title_match = re.search(r'(?:sermon|message|talk)(?:\s+(?:about|on|titled|called|named))?\s+["\']?([^"\'?.]+)["\']?', query, re.IGNORECASE)
        title_filter = None
        if title_match:
            title_filter = title_match.group(1).strip()
            # Remove the title reference from the query for better embedding
            processed_query = re.sub(r'(?:sermon|message|talk)(?:\s+(?:about|on|titled|called|named))?\s+["\']?([^"\'?.]+)["\']?', '', processed_query, flags=re.IGNORECASE).strip()
        
        # Build filter for Pinecone
        pinecone_filter = {}
        if date_filter:
            # Use approximate date matching (within 48 hours)
            day_start = date_filter - (48 * 60 * 60)  # Subtract two days in seconds
            day_end = date_filter + (48 * 60 * 60)    # Add two days in seconds
            pinecone_filter["publish_date"] = {"$gte": day_start, "$lte": day_end}
            
        # Translate query to English if needed
        english_query = processed_query
        if needs_translation:
            english_query = await translate_text(processed_query, original_language, "en")
            print(f"Translated query from {original_language} to English: {english_query}")
        
        # Generate embedding for the translated query
        query_embedding = generate_embedding(english_query)
        
        # Search Pinecone with filters
        search_response = pinecone_index.query(
            vector=query_embedding,
            top_k=request.top_k,
            include_metadata=True,
            filter=pinecone_filter if pinecone_filter else None
        )
        
        # Format search results 
        search_results = []
        
        for match in search_response.matches:
            if match.score < 0.5:  # Minimum threshold for relevance
                continue
                
            metadata = match.metadata
            video_id = metadata.get("video_id", "")
            
            # Load additional metadata to get proper title and date
            enhanced_metadata = load_metadata(video_id)
            
            # Apply title filter if specified
            if title_filter and title_filter.lower() not in enhanced_metadata.get("title", "").lower():
                continue
            
            # Validate and get proper date
            publish_date = get_proper_date_from_metadata(enhanced_metadata)
            
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
                publish_date=publish_date
            ))
        
        # Generate suggested queries (both for no results and to enhance results)
        suggested_queries = []
        if len(search_results) == 0:
            # Get suggestions with sample results for no-results case
            suggested_queries, sample_results = generate_suggested_queries_with_results(query)
        else:
            # Still get some related queries for normal results
            suggested_queries = generate_suggested_queries(query, max_suggestions=2)
        
        # Generate answer text with integrated suggestions
        if search_results:
            # Normal case - we have search results
            answer_text = generate_enhanced_ai_answer(english_query, search_results, "en")
            
            # Add suggestions directly to the answer text for better UX
            if suggested_queries:
                answer_text = format_response_with_suggestions(answer_text, suggested_queries)
        else:
            # No results case
            no_results_message = generate_no_results_message(query, suggested_queries, "en")
            answer_text = format_response_with_suggestions(no_results_message, suggested_queries)
            
            # If we had sample results, include the first one
            if 'sample_results' in locals() and sample_results:
                search_results.append(sample_results[0])
        
        # Translate to the requested language if needed
        if needs_translation:
            answer_text = await translate_text(answer_text, "en", original_language)
            print(f"Translated answer from English to {original_language}")
        
        return AnswerResponse(
            query=request.query,  # Return the original untranslated query
            answer=answer_text,
            sources=search_results if request.include_sources else [],
            processing_time=time.time() - start_time,
            suggested_queries=suggested_queries
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
    offset: int = Query(0, description="Offset for pagination")
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
        sermon_list.sort(key=lambda x: x.get("publish_date", ""), reverse=True)
        
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