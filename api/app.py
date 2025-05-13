import os
import time
import glob
import json
import re
from typing import List, Dict, Optional, Any, Union
from datetime import datetime, timedelta
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
QUERY_ENHANCEMENT_MODEL = os.environ.get("QUERY_ENHANCEMENT_MODEL", "gpt-4o") # Model for query enhancement

# Path to metadata directory
METADATA_DIR = os.environ.get("METADATA_DIR", "./transcription/data/metadata")
SUBTITLES_DIR = os.environ.get("SUBTITLES_DIR", "./transcription/data/subtitles")
# Add Bible references directory path
BIBLE_REFERENCES_DIR = os.environ.get("BIBLE_REFERENCES_DIR", "./transcription/data/bible_references")
# Cache directory
CACHE_DIR = os.environ.get("CACHE_DIR", "./cache")

# Create cache directory if it doesn't exist
os.makedirs(CACHE_DIR, exist_ok=True)

# Feature flags (can be enabled/disabled via environment variables)
ENABLE_QUERY_REFINEMENT = os.environ.get("ENABLE_QUERY_REFINEMENT", "true").lower() == "true"
ENABLE_SERMON_CONTEXT = os.environ.get("ENABLE_SERMON_CONTEXT", "true").lower() == "true"
ENABLE_RESPONSE_CACHING = os.environ.get("ENABLE_RESPONSE_CACHING", "true").lower() == "true"
ENABLE_THEMATIC_ANALYSIS = os.environ.get("ENABLE_THEMATIC_ANALYSIS", "true").lower() == "true"
CACHE_TTL_HOURS = int(os.environ.get("CACHE_TTL_HOURS", "24"))

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
    # New field - don't break clients that don't expect it
    sermon_context: Optional[str] = None

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total_results: int
    processing_time: float
    # New fields - don't break clients that don't expect them
    improved_query: Optional[str] = None
    suggested_queries: Optional[List[str]] = None
    thematic_summary: Optional[str] = None

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
    # New fields - don't break clients that don't expect them
    original_query: Optional[str] = None
    improved_query: Optional[str] = None
    suggested_followup_questions: Optional[List[str]] = None

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

# Cache implementation
class ResponseCache:
    def __init__(self, cache_dir, ttl_hours=24):
        self.cache_dir = cache_dir
        self.ttl_hours = ttl_hours
        
    def _get_cache_key(self, endpoint, params):
        """Create a standardized cache key from endpoint and parameters."""
        # Sort params to ensure consistent caching regardless of order
        param_str = json.dumps(params, sort_keys=True)
        import hashlib
        # Create hash to use as filename
        hash_key = hashlib.md5(param_str.encode()).hexdigest()
        return f"{endpoint}_{hash_key}"
    
    def _get_cache_file_path(self, cache_key):
        """Get the file path for a cache key."""
        return os.path.join(self.cache_dir, f"{cache_key}.json")
    
    def get(self, endpoint, params):
        """Retrieve cached response if it exists and hasn't expired."""
        if not ENABLE_RESPONSE_CACHING:
            return None
            
        cache_key = self._get_cache_key(endpoint, params)
        cache_file = self._get_cache_file_path(cache_key)
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                
                # Check if cache has expired
                cache_time = datetime.fromisoformat(cache_data.get('timestamp'))
                if datetime.now() - cache_time < timedelta(hours=self.ttl_hours):
                    return cache_data.get('response')
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"Cache read error: {str(e)}")
        
        return None
    
    def set(self, endpoint, params, response):
        """Cache a response with current timestamp."""
        if not ENABLE_RESPONSE_CACHING:
            return
            
        cache_key = self._get_cache_key(endpoint, params)
        cache_file = self._get_cache_file_path(cache_key)
        
        try:
            cache_data = {
                'timestamp': datetime.now().isoformat(),
                'response': response
            }
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Cache write error: {str(e)}")
    
    def invalidate(self, endpoint=None):
        """Invalidate cache for specified endpoint or all if None."""
        try:
            cache_files = glob.glob(os.path.join(self.cache_dir, f"{'*' if endpoint is None else endpoint+'_'}*.json"))
            for file in cache_files:
                os.remove(file)
            return len(cache_files)
        except Exception as e:
            print(f"Cache invalidation error: {str(e)}")
            return 0

# Initialize cache
response_cache = ResponseCache(CACHE_DIR, CACHE_TTL_HOURS)

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

# NEW: Enhanced query functions
def enhance_query(original_query: str) -> Dict[str, Any]:
    """
    Enhance an original query to improve search results.
    Returns improved query and suggested alternative queries.
    """
    if not ENABLE_QUERY_REFINEMENT:
        return {"improved_query": original_query, "suggested_queries": []}
        
    try:
        prompt = f"""
You are an AI assistant specialized in helping users find relevant sermon content. Your task is to:

1. Analyze the user's search query
2. Generate an improved version that will give better semantic search results
3. Suggest 2-3 alternative queries that might help the user find what they're looking for

Original query: "{original_query}"

Respond in JSON format with these fields:
- improved_query: An enhanced version of the original query
- suggested_queries: Array of 2-3 alternative queries
- rationale: Brief explanation of your improvements

Focus on Biblical terms, theological concepts, and sermon-specific language.
"""
        
        response = openai_client.chat.completions.create(
            model=QUERY_ENHANCEMENT_MODEL,
            messages=[
                {"role": "system", "content": "You help improve search queries for sermon content."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=500,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return {
            "improved_query": result.get("improved_query", original_query),
            "suggested_queries": result.get("suggested_queries", []),
            "rationale": result.get("rationale", "")
        }
    except Exception as e:
        print(f"Query enhancement error: {str(e)}")
        # On error, return original query with no suggestions
        return {"improved_query": original_query, "suggested_queries": []}

def generate_followup_questions(query: str, answer: str) -> List[str]:
    """Generate follow-up questions based on the query and answer."""
    if not ENABLE_QUERY_REFINEMENT:
        return []
        
    try:
        prompt = f"""
Based on the following user query and the answer provided, generate 3 natural follow-up questions the user might want to ask next.

USER QUERY: {query}

ANSWER: {answer}

Generate 3 specific, contextually relevant follow-up questions that:
1. Dig deeper into aspects mentioned in the answer
2. Clarify theological concepts in the answer
3. Explore related biblical themes or passages

FOLLOW-UP QUESTIONS (only list the questions, no explanations):
"""
        
        response = openai_client.chat.completions.create(
            model=QUERY_ENHANCEMENT_MODEL,
            messages=[
                {"role": "system", "content": "You help generate relevant follow-up questions about sermon content."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=250
        )
        
        # Extract questions (one per line)
        raw_response = response.choices[0].message.content.strip()
        questions = []
        
        # Parse numbered lists like "1. Question", bullet points, or plain lines
        for line in raw_response.split('\n'):
            # Remove numbers or bullet points
            cleaned_line = re.sub(r'^(\d+\.|\*|\-)\s*', '', line.strip())
            if cleaned_line and len(cleaned_line) > 10:  # Minimum length to be a real question
                questions.append(cleaned_line)
                
        return questions[:3]  # Limit to 3 questions
    except Exception as e:
        print(f"Follow-up questions generation error: {str(e)}")
        return []

# NEW: Sermon context enrichment
def generate_sermon_context(results: List[SearchResult]) -> Dict[str, str]:
    """Generate rich context for each sermon to enhance understanding."""
    if not ENABLE_SERMON_CONTEXT:
        return {}
        
    # Group results by sermon (video_id)
    sermons_by_id = {}
    for result in results:
        if result.video_id not in sermons_by_id:
            sermons_by_id[result.video_id] = {
                "title": result.title,
                "publish_date": result.publish_date,
                "segments": []
            }
        sermons_by_id[result.video_id]["segments"].append(result.text)
    
    # Generate context for each sermon
    sermon_contexts = {}
    for video_id, sermon_data in sermons_by_id.items():
        try:
            # Get combined text from all segments for this sermon (limited to avoid token limits)
            combined_text = " ".join(sermon_data["segments"])
            if len(combined_text) > 6000:
                combined_text = combined_text[:6000] + "..."
                
            prompt = f"""
Analyze this sermon segment and provide a brief context that would help the user better understand it.

SERMON TITLE: {sermon_data["title"]}
PUBLISH DATE: {sermon_data["publish_date"] if sermon_data["publish_date"] else "Unknown"}

SERMON SEGMENT:
{combined_text}

Provide a VERY BRIEF (30-50 words max) contextual summary that:
1. Identifies the main theological theme or Biblical passage being discussed
2. Notes any relevant context (holiday, church calendar, current events at the time)
3. Highlights key teaching points

BRIEF CONTEXT:
"""
            
            response = openai_client.chat.completions.create(
                model=COMPLETION_MODEL,
                messages=[
                    {"role": "system", "content": "You provide brief, insightful context for sermon segments."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=80  # Keep it very brief
            )
            
            sermon_contexts[video_id] = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Sermon context generation error for {video_id}: {str(e)}")
            sermon_contexts[video_id] = f"Sermon about {sermon_data['title']}"
            
    return sermon_contexts

# NEW: Thematic analysis
def generate_thematic_summary(results: List[SearchResult], query: str) -> str:
    """Generate a thematic summary of search results."""
    if not ENABLE_THEMATIC_ANALYSIS or len(results) < 2:
        return None
        
    try:
        # Combine relevant segments (limited to avoid token limits)
        segments_text = "\n\n".join([
            f"SEGMENT {i+1} (From sermon: {r.title}):\n{r.text[:300]}..." 
            for i, r in enumerate(results[:5])
        ])
        
        prompt = f"""
Analyze these sermon segments related to the search query "{query}" and provide a brief thematic summary.

SERMON SEGMENTS:
{segments_text}

Provide a BRIEF (50 words max) summary that:
1. Identifies common theological themes or Biblical passages across the segments
2. Notes any different perspectives or approaches to the topic
3. Highlights key teaching points that appear consistently

THEMATIC SUMMARY:
"""
        
        response = openai_client.chat.completions.create(
            model=COMPLETION_MODEL,
            messages=[
                {"role": "system", "content": "You provide insightful thematic summaries of sermon content."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=100
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Thematic summary generation error: {str(e)}")
        return None

def generate_ai_answer(query: str, search_results: List[SearchResult], language: str = "en") -> str:
    """Generate an AI answer based on the search results in the specified language."""
    # Prepare the context from search results
    context = "\n\n".join([
        f"Segment {i+1} (Time: {format_time(result.start_time)} - {format_time(result.end_time)}):\n{result.text}"
        for i, result in enumerate(search_results)
    ])
    
    # Add sermon context as additional information if available
    sermon_context = ""
    if ENABLE_SERMON_CONTEXT and any(result.sermon_context for result in search_results):
        sermon_context = "\n\nADDITIONAL SERMON CONTEXT:\n" + "\n".join([
            f"- {result.title}: {result.sermon_context}" 
            for result in search_results if result.sermon_context
        ])
    
    # Set system message based on language
    if language == "es":
        system_message = "Eres un asistente que ayuda a los usuarios a entender el contenido de sermones. Responde en español."
    elif language == "zh":
        system_message = "你是一个帮助用户理解讲道内容的助手。用中文回答。"
    else:
        system_message = "You are a helpful assistant that answers questions about sermon content."
    
    # Prepare the prompt for GPT-4 based on language
    if language == "es":
        prompt = f"""
Responde a la siguiente pregunta basándote únicamente en los segmentos de sermón proporcionados. Si la respuesta no se encuentra en los segmentos, dilo claramente.

PREGUNTA DEL USUARIO: {query}

SEGMENTOS DEL SERMÓN:
{context}
{sermon_context}

Responde a la pregunta basándote únicamente en los segmentos de sermón proporcionados. Incluye referencias específicas a qué segmento(s) contienen la información (por ejemplo, "En el Segmento 3, el pastor explica..."). Mantén tu respuesta enfocada y concisa.
        """
    elif language == "zh":
        prompt = f"""
根据提供的讲道片段回答以下问题。如果在这些片段中找不到答案，请清楚地说明。

用户问题: {query}

讲道片段:
{context}
{sermon_context}

仅根据提供的讲道片段回答问题。包括具体引用哪个片段包含信息（例如，"在片段3中，牧师解释了..."）。保持回答重点明确和简洁。
        """
    else:
        prompt = f"""
Answer the following question based only on the provided sermon segments. If the answer cannot be found in the segments, say so clearly.

USER QUESTION: {query}

SERMON SEGMENTS:
{context}
{sermon_context}

Answer the question based only on the provided sermon segments. Include specific references to which segment(s) contain the information (e.g., "In Segment 3, the pastor explains..."). Keep your response focused and concise.
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
        
        # Check response cache
        cache_files = len(glob.glob(os.path.join(CACHE_DIR, "*.json")))
        
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
            "cache": {
                "status": "enabled" if ENABLE_RESPONSE_CACHING else "disabled",
                "file_count": cache_files,
                "ttl_hours": CACHE_TTL_HOURS
            },
            "features": {
                "query_refinement": ENABLE_QUERY_REFINEMENT,
                "sermon_context": ENABLE_SERMON_CONTEXT,
                "response_caching": ENABLE_RESPONSE_CACHING,
                "thematic_analysis": ENABLE_THEMATIC_ANALYSIS
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
    min_score: float = Query(0.6, description="Minimum similarity score (0-1)"),
    enhance: bool = Query(False, description="Whether to enhance the query")
):
    """
    Search for sermon segments matching the query.
    Returns the most semantically similar segments from the sermon library.
    """
    start_time = time.time()
    
    # First check cache if enabled
    cache_key = "search"
    cache_params = {"query": query, "top_k": top_k, "min_score": min_score, "enhance": enhance}
    cached_response = response_cache.get(cache_key, cache_params)
    
    if cached_response:
        # Add processing time and return cached response
        cached_response["processing_time"] = time.time() - start_time
        return cached_response
    
    # Query enhancement (if enabled and requested)
    original_query = query
    suggested_queries = []
    
    if ENABLE_QUERY_REFINEMENT and enhance:
        enhancement_result = enhance_query(query)
        query = enhancement_result["improved_query"]
        suggested_queries = enhancement_result["suggested_queries"]
    
    try:
        # Generate embedding for the query
        query_embedding = generate_embedding(query)
        
        # Search Pinecone - updated for v6.0.2 API
        search_response = pinecone_index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
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
        
        # Generate sermon contexts if enabled
        if ENABLE_SERMON_CONTEXT and results:
            sermon_contexts = generate_sermon_context(results)
            # Add context to each result
            for result in results:
                if result.video_id in sermon_contexts:
                    result.sermon_context = sermon_contexts[result.video_id]
        
        # Generate thematic summary if enabled and we have enough results
        thematic_summary = None
        if ENABLE_THEMATIC_ANALYSIS and len(results) >= 2:
            thematic_summary = generate_thematic_summary(results, query)
        
        # Create response
        response = SearchResponse(
            query=query,
            results=results,
            total_results=len(results),
            processing_time=time.time() - start_time,
            improved_query=query if query != original_query else None,
            suggested_queries=suggested_queries if suggested_queries else None,
            thematic_summary=thematic_summary
        )
        
        # Cache the response if enabled
        if ENABLE_RESPONSE_CACHING:
            response_cache.set(cache_key, cache_params, response.dict())
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

@app.post("/answer", response_model=AnswerResponse)
async def answer(request: AnswerRequest):
    """
    Generate an AI answer to a question based on sermon content.
    Searches for relevant sermon segments and uses them to create a response.
    """
    start_time = time.time()
    
    # First check cache if enabled
    cache_key = "answer"
    cache_params = request.dict()
    cached_response = response_cache.get(cache_key, cache_params)
    
    if cached_response:
        # Add processing time and return cached response
        cached_response["processing_time"] = time.time() - start_time
        return cached_response
    
    try:
        # Store original query for reference
        original_query = request.query
        
        # Optional query enhancement
        improved_query = original_query
        if ENABLE_QUERY_REFINEMENT:
            enhancement_result = enhance_query(original_query)
            improved_query = enhancement_result["improved_query"]
        
        # Determine if we need to translate the query
        original_language = request.language
        needs_translation = original_language != "en"
        
        # Translate query to English if needed
        query_for_search = improved_query
        if needs_translation:
            query_for_search = await translate_text(improved_query, original_language, "en")
            print(f"Translated query from {original_language} to English: {query_for_search}")
        
        # Generate embedding for the translated query
        query_embedding = generate_embedding(query_for_search)
        
        # Search Pinecone - updated for v6.0.2 API
        search_response = pinecone_index.query(
            vector=query_embedding,
            top_k=request.top_k,
            include_metadata=True
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
        
        # Generate sermon contexts if enabled
        if ENABLE_SERMON_CONTEXT and search_results:
            sermon_contexts = generate_sermon_context(search_results)
            # Add context to each result
            for result in search_results:
                if result.video_id in sermon_contexts:
                    result.sermon_context = sermon_contexts[result.video_id]
        
        # Generate AI answer - always generate in English first, then translate if needed
        default_no_results = "No relevant sermon content found to answer this question."
        
        if search_results:
            # Always generate in English first for consistency
            answer_text = generate_ai_answer(query_for_search, search_results, "en")
            
            # Then translate to the requested language if needed
            if needs_translation:
                answer_text = await translate_text(answer_text, "en", original_language)
                print(f"Translated answer from English to {original_language}")
        else:
            # Handle no results case with appropriate translation
            answer_text = default_no_results
            if needs_translation:
                answer_text = await translate_text(default_no_results, "en", original_language)
        
        # Generate follow-up questions if enabled
        followup_questions = None
        if ENABLE_QUERY_REFINEMENT and search_results:
            followup_questions = generate_followup_questions(request.query, answer_text)
        
        # Create response
        response = AnswerResponse(
            query=request.query,  # Return the original untranslated query
            answer=answer_text,
            sources=search_results if request.include_sources else [],
            processing_time=time.time() - start_time,
            original_query=original_query if improved_query != original_query else None,
            improved_query=improved_query if improved_query != original_query else None,
            suggested_followup_questions=followup_questions
        )
        
        # Cache the response if enabled
        if ENABLE_RESPONSE_CACHING:
            response_cache.set(cache_key, cache_params, response.dict())
        
        return response
        
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
        
        # If sermon context is enabled, generate and add context
        sermon_context = None
        if ENABLE_SERMON_CONTEXT:
            try:
                # Create a mock SearchResult to generate context
                mock_result = SearchResult(
                    video_id=video_id,
                    title=enhanced_metadata.get("title", f"Sermon {video_id}"),
                    url=f"https://www.youtube.com/watch?v={video_id}",
                    text="\n".join([s["text"] for s in processed_segments[:5] if "is_gap" not in s]),
                    start_time=processed_segments[0]["start_time"] if processed_segments else 0,
                    end_time=processed_segments[-1]["end_time"] if processed_segments else 0,
                    similarity=1.0,
                    chunk_index=0,
                    publish_date=enhanced_metadata.get("publish_date")
                )
                
                sermon_contexts = generate_sermon_context([mock_result])
                if video_id in sermon_contexts:
                    sermon_context = sermon_contexts[video_id]
            except Exception as e:
                print(f"Error generating sermon context: {str(e)}")
        
        response = {
            "video_id": video_id,
            "title": enhanced_metadata.get("title", f"Sermon {video_id}"),
            "publish_date": enhanced_metadata.get("publish_date"),
            "language": language,
            "segments": processed_segments,
            "total_segments": len(processed_segments),
            "transcript_source": "pinecone"
        }
        
        # Add sermon context if available (won't break existing clients)
        if sermon_context:
            response["sermon_context"] = sermon_context
        
        return response
        
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
        if len(all_sermons) < 400:  # Assuming we should have at least 400 sermons based on count
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
        
        # If sermon context is enabled, generate and add it
        sermon_context = None
        if ENABLE_SERMON_CONTEXT and chunks:
            try:
                # Create a mock SearchResult to generate context
                combined_text = " ".join([chunk["text"] for chunk in chunks[:5]])
                if len(combined_text) > 4000:
                    combined_text = combined_text[:4000] + "..."
                    
                mock_result = SearchResult(
                    video_id=video_id,
                    title=sermon_info["title"],
                    url=sermon_info["url"],
                    text=combined_text,
                    start_time=chunks[0]["start_time"] if chunks else 0,
                    end_time=chunks[-1]["end_time"] if chunks else 0,
                    similarity=1.0,
                    chunk_index=0,
                    publish_date=sermon_info.get("publish_date")
                )
                
                sermon_contexts = generate_sermon_context([mock_result])
                if video_id in sermon_contexts:
                    sermon_context = sermon_contexts[video_id]
            except Exception as e:
                print(f"Error generating sermon context: {str(e)}")
        
        # Create response
        response = {
            "sermon": sermon_info,
            "chunks": chunks,
            "total_chunks": len(chunks)
        }
        
        # Add sermon context if available (won't break existing clients)
        if sermon_context:
            response["sermon"]["context"] = sermon_context
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving sermon: {str(e)}")

@app.post("/cache/invalidate")
async def invalidate_cache(endpoint: Optional[str] = None):
    """Invalidate cached responses for an endpoint or all endpoints."""
    if not ENABLE_RESPONSE_CACHING:
        return {"status": "error", "message": "Caching is disabled"}
        
    count = response_cache.invalidate(endpoint)
    return {
        "status": "success", 
        "message": f"Invalidated {count} cached {'responses' if count != 1 else 'response'}",
        "endpoint": endpoint if endpoint else "all"
    }

# Bible reference endpoints
@app.get("/bible/stats", response_model=BibleReferenceStats)
async def get_bible_reference_stats():
    """Get statistics about Bible references in sermons."""
    try:
        # Check cache first
        cache_key = "bible_stats"
        cached_stats = response_cache.get(cache_key, {})
        
        if cached_stats:
            return cached_stats
            
        stats = get_bible_stats()
        
        # Cache the result
        if ENABLE_RESPONSE_CACHING:
            response_cache.set(cache_key, {}, stats.dict())
            
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
        
        # Create response
        response = {
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
        
        # If thematic analysis is enabled, add thematic summary
        if ENABLE_THEMATIC_ANALYSIS and len(enhanced_references) >= 3:
            try:
                # Extract contexts from references to analyze themes
                contexts = [ref.get("context", "") for ref in enhanced_references if ref.get("context")]
                if contexts:
                    prompt = f"""
                    Analyze these {len(contexts)} sermon segments that reference {display_text}. 
                    Provide a brief (30-50 words) summary of how this scripture is typically used or interpreted 
                    in these sermons. Focus on theological themes and applications.
                    
                    CONTEXTS:
                    {' '.join(contexts[:5])}
                    
                    THEMATIC SUMMARY:
                    """
                    
                    response_data = openai_client.chat.completions.create(
                        model=COMPLETION_MODEL,
                        messages=[
                            {"role": "system", "content": "You summarize how Bible verses are used in sermons."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.4,
                        max_tokens=100
                    )
                    
                    thematic_summary = response_data.choices[0].message.content.strip()
                    response["thematic_summary"] = thematic_summary
            except Exception as e:
                print(f"Error generating thematic summary: {str(e)}")
        
        return response
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving reference: {str(e)}")

# NEW: Query suggestions endpoint - provides suggestions when no results found
@app.get("/query/suggest")
async def suggest_query(
    query: str = Query(..., description="The search query that returned no results"),
    context: str = Query("sermon", description="Context for suggestions (sermon, bible)")
):
    """
    Generate suggested queries when a search returns no results.
    Returns alternative queries that might yield better results.
    """
    if not ENABLE_QUERY_REFINEMENT:
        return {
            "original_query": query,
            "suggested_queries": [],
            "message": "Query refinement is disabled"
        }
        
    try:
        # First check cache
        cache_key = "query_suggest"
        cache_params = {"query": query, "context": context}
        cached_response = response_cache.get(cache_key, cache_params)
        
        if cached_response:
            return cached_response
            
        # Define prompt based on context
        if context.lower() == "bible":
            system_prompt = "You help users find relevant Bible passages when their searches don't return results."
            prompt = f"""
            The user searched for "{query}" in relation to Bible references but found no results.
            
            Generate 5 alternative search queries that:
            1. Use more standard Biblical terminology or phrasing
            2. Include common Bible verse references related to this topic
            3. Use theological terms that might appear in sermons discussing this topic
            
            For each suggestion, briefly explain why it might yield better results.
            
            Format your response as JSON with:
            - suggested_queries: array of alternative queries (strings)
            - explanations: matching array of explanations for each query
            """
        else:  # Default sermon context
            system_prompt = "You help users find relevant sermon content when their searches don't return results."
            prompt = f"""
            The user searched for "{query}" in our sermon database but found no results.
            
            Generate 5 alternative search queries that:
            1. Use more common theological or Biblical terminology 
            2. Broaden or narrow the scope appropriately
            3. Reframe the concept using language likely found in sermons
            
            For each suggestion, briefly explain why it might yield better results.
            
            Format your response as JSON with:
            - suggested_queries: array of alternative queries (strings)
            - explanations: matching array of explanations for each query
            """
        
        response = openai_client.chat.completions.create(
            model=QUERY_ENHANCEMENT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # Format response
        response_data = {
            "original_query": query,
            "suggested_queries": result.get("suggested_queries", [])[:5],
            "explanations": result.get("explanations", [])[:5],
            "context": context
        }
        
        # Cache the response
        if ENABLE_RESPONSE_CACHING:
            response_cache.set(cache_key, cache_params, response_data)
            
        return response_data
        
    except Exception as e:
        print(f"Query suggestion error: {str(e)}")
        return {
            "original_query": query,
            "suggested_queries": [],
            "error": "Failed to generate suggestions"
        }

# NEW: Thematic exploration endpoint - discover common themes across sermons
@app.get("/themes")
async def explore_themes(
    topic: Optional[str] = Query(None, description="Topic to explore (optional)"),
    limit: int = Query(5, description="Number of themes to return")
):
    """
    Discover and explore common themes across sermons.
    Returns thematic categories and representative sermons.
    """
    if not ENABLE_THEMATIC_ANALYSIS:
        return {
            "message": "Thematic analysis is disabled",
            "themes": []
        }
        
    try:
        # First check cache
        cache_key = "themes"
        cache_params = {"topic": topic, "limit": limit}
        cached_response = response_cache.get(cache_key, cache_params)
        
        if cached_response:
            return cached_response
            
        # If topic provided, search for relevant sermons
        search_results = []
        if topic:
            # Generate embedding for the topic
            query_embedding = generate_embedding(topic)
            
            # Search Pinecone
            search_response = pinecone_index.query(
                vector=query_embedding,
                top_k=min(limit * 3, 20),  # Get more results to ensure diversity
                include_metadata=True
            )
            
            # Format results
            for match in search_response.matches:
                if match.score < 0.5:  # Minimum threshold for relevance
                    continue
                    
                metadata = match.metadata
                video_id = metadata.get("video_id", "")
                
                # Load additional metadata to get proper title and date
                enhanced_metadata = load_metadata(video_id)
                
                search_results.append({
                    "video_id": video_id,
                    "title": enhanced_metadata.get("title", metadata.get("title", "Unknown Sermon")),
                    "text": metadata.get("text", ""),
                    "similarity": match.score
                })
        else:
            # If no topic, get random sermons from different time periods
            # This is a simplified approach - in production, you would want a more sophisticated sampling
            try:
                # Get index stats to confirm vector count
                stats = pinecone_index.describe_index_stats()
                total_vectors = stats.total_vector_count
                
                # Use a few different random vectors to get diversity
                for i in range(5):
                    # Create different random vectors
                    random_vector = [(i * 2 + j) / 1000.0 for j in range(1536)]
                    
                    response = pinecone_index.query(
                        vector=random_vector,
                        top_k=5,
                        include_metadata=True
                    )
                    
                    # Process each match
                    for match in response.matches:
                        metadata = match.metadata
                        video_id = metadata.get("video_id", "")
                        
                        # Skip duplicates
                        if any(r["video_id"] == video_id for r in search_results):
                            continue
                            
                        # Load enhanced metadata
                        enhanced_metadata = load_metadata(video_id)
                        
                        search_results.append({
                            "video_id": video_id,
                            "title": enhanced_metadata.get("title", metadata.get("title", "Unknown Sermon")),
                            "text": metadata.get("text", ""),
                            "similarity": match.score
                        })
                        
                        # Limit to needed amount
                        if len(search_results) >= limit * 3:
                            break
                    
                    if len(search_results) >= limit * 3:
                        break
                        
            except Exception as e:
                print(f"Error getting random sermons: {str(e)}")
                # Fallback to topic search if random approach fails
                if not topic:
                    topic = "faith"  # Default topic if none provided
                    query_embedding = generate_embedding(topic)
                    
                    search_response = pinecone_index.query(
                        vector=query_embedding,
                        top_k=min(limit * 3, 20),
                        include_metadata=True
                    )
                    
                    for match in search_response.matches:
                        metadata = match.metadata
                        video_id = metadata.get("video_id", "")
                        
                        # Load additional metadata
                        enhanced_metadata = load_metadata(video_id)
                        
                        search_results.append({
                            "video_id": video_id,
                            "title": enhanced_metadata.get("title", metadata.get("title", "Unknown Sermon")),
                            "text": metadata.get("text", ""),
                            "similarity": match.score
                        })
        
        # If we have sermon results, analyze them for themes
        if search_results:
            # Combine sermon texts for analysis (limit to avoid token limits)
            sermon_texts = []
            for result in search_results[:15]:  # Limit to 15 for token reasons
                # Limit individual text length
                text = result["text"]
                if len(text) > 1000:
                    text = text[:1000] + "..."
                sermon_texts.append({
                    "title": result["title"],
                    "text": text,
                    "video_id": result["video_id"]
                })
            
            # Generate thematic analysis
            prompt = f"""
            Analyze these sermon excerpts and identify {limit} distinct theological or biblical themes.
            
            {json.dumps(sermon_texts, indent=2)}
            
            For each theme:
            1. Provide a concise name (e.g., "Grace through Faith", "Spiritual Disciplines")
            2. Write a brief (1-2 sentence) description of the theme
            3. List 1-3 sermon titles from the provided sermons that best represent this theme
            4. List 1-3 Bible references commonly associated with this theme
            
            Format your response as JSON with this structure:
            {{
                "themes": [
                    {{
                        "name": "Theme name",
                        "description": "Theme description",
                        "representative_sermons": ["sermon title 1", "sermon title 2"],
                        "video_ids": ["video_id1", "video_id2"],
                        "bible_references": ["John 3:16", "Romans 8:28"]
                    }}
                ]
            }}
            
            Return exactly {limit} themes, focusing on quality and diversity.
            """
            
            response = openai_client.chat.completions.create(
                model=COMPLETION_MODEL,
                messages=[
                    {"role": "system", "content": "You analyze sermon content to identify theological themes."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1500,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            themes = result.get("themes", [])
            
            # Match video_ids to sermon titles if not already provided
            for theme in themes:
                if "video_ids" not in theme or not theme["video_ids"]:
                    theme["video_ids"] = []
                    for sermon_title in theme.get("representative_sermons", []):
                        # Find matching sermon from our results
                        for result in search_results:
                            if sermon_title.lower() in result["title"].lower():
                                if result["video_id"] not in theme["video_ids"]:
                                    theme["video_ids"].append(result["video_id"])
                                break
            
            response_data = {
                "query": topic,
                "themes": themes,
                "count": len(themes)
            }
            
            # Cache the response
            if ENABLE_RESPONSE_CACHING:
                response_cache.set(cache_key, cache_params, response_data)
                
            return response_data
        else:
            return {
                "query": topic,
                "themes": [],
                "message": "No sermon content found to analyze"
            }
            
    except Exception as e:
        print(f"Theme exploration error: {str(e)}")
        return {
            "query": topic,
            "themes": [],
            "error": "Failed to generate themes"
        }

# NEW: Related sermons endpoint - find similar sermons
@app.get("/sermons/{video_id}/related")
async def get_related_sermons(
    video_id: str,
    limit: int = Query(5, description="Number of related sermons to return")
):
    """
    Find sermons related to a specific sermon.
    Returns a list of similar sermons based on content similarity.
    """
    try:
        # First check cache
        cache_key = "related_sermons"
        cache_params = {"video_id": video_id, "limit": limit}
        cached_response = response_cache.get(cache_key, cache_params)
        
        if cached_response:
            return cached_response
            
        # Load metadata for current sermon
        enhanced_metadata = load_metadata(video_id)
        
        # Create a vector that represents this sermon
        # First try to get the chunks for this sermon
        filter_dict = {"video_id": {"$eq": video_id}}
        
        # Get chunks for this sermon
        chunks_response = pinecone_index.query(
            vector=[0.1] * 1536,  # Placeholder vector
            top_k=20,  # Get several chunks to build a good representation
            include_metadata=True,
            filter=filter_dict
        )
        
        if not chunks_response.matches:
            raise HTTPException(status_code=404, detail=f"Sermon not found: {video_id}")
        
        # Get a sampling of chunks from this sermon
        chunks = []
        for match in chunks_response.matches:
            metadata = match.metadata
            if metadata.get("video_id") == video_id and metadata.get("text"):
                chunks.append(metadata.get("text", ""))
                if len(chunks) >= 10:  # Limit to 10 chunks
                    break
        
        # Combine chunks into a query for related content
        sermon_text = " ".join(chunks)
        if len(sermon_text) > 8000:
            sermon_text = sermon_text[:8000]  # Truncate to avoid token limits
        
        # Create an embedding for the combined text
        query_embedding = generate_embedding(sermon_text)
        
        # Search for related sermons, excluding the current one
        search_response = pinecone_index.query(
            vector=query_embedding,
            top_k=limit * 3,  # Get more to ensure we have enough unique sermons
            include_metadata=True
        )
        
        # Process results - group by sermon and exclude the current sermon
        related_sermons = {}
        
        for match in search_response.matches:
            metadata = match.metadata
            related_video_id = metadata.get("video_id", "")
            
            # Skip if this is the same sermon or we already have it
            if related_video_id == video_id or related_video_id in related_sermons:
                continue
                
            # Load metadata for related sermon
            related_metadata = load_metadata(related_video_id)
            
            related_sermons[related_video_id] = {
                "video_id": related_video_id,
                "title": related_metadata.get("title", metadata.get("title", "Unknown Sermon")),
                "similarity": match.score,
                "url": f"https://www.youtube.com/watch?v={related_video_id}",
                "publish_date": related_metadata.get("publish_date"),
                "sample_text": metadata.get("text", "")[:200] + "..."  # Preview text
            }
            
            # Stop once we have enough
            if len(related_sermons) >= limit:
                break
        
        # Convert to list and sort by similarity
        related_list = list(related_sermons.values())
        related_list.sort(key=lambda x: x["similarity"], reverse=True)
        
        # If we have enough sermons, try to identify common themes
        common_themes = None
        # Fixed version of the problematic code block
        if ENABLE_THEMATIC_ANALYSIS and len(related_list) >= 3:
            try:
                # Get text samples from related sermons - FIXED: Don't use f-strings with backslashes
                sermon_samples = []
                for s in related_list[:5]:
                    sermon_samples.append("Sermon: {}\nText: {}".format(
                        s['title'], 
                        s['sample_text']
                    ))
                
                # Add original sermon - FIXED: Don't use nested f-strings
                original_title = enhanced_metadata.get('title', 'Sermon {}'.format(video_id))
                original_text = chunks[0] if chunks else ''
                sermon_samples.insert(0, "Original Sermon: {}\nText: {}".format(original_title, original_text))
                
                # FIXED: Don't use f-string with JSON examples and escape sequences
                prompt = """
                Analyze these related sermon excerpts and identify 2-3 common theological or biblical themes that connect them.
                
                {}
                
                For each theme:
                1. Provide a concise name (e.g., "Grace through Faith")
                2. Write a brief (1-2 sentence) description of why this theme connects these sermons
                
                Format your response as JSON with this structure:
                {{
                    "common_themes": [
                        {{
                            "name": "Theme name",
                            "description": "Theme description"
                        }}
                    ]
                }}
                """.format('\n\n'.join(sermon_samples))
                
                response = openai_client.chat.completions.create(
                    model=COMPLETION_MODEL,
                    messages=[
                        {"role": "system", "content": "You identify common themes between related sermons."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.5,
                    max_tokens=500,
                    response_format={"type": "json_object"}
                )
                
                result = json.loads(response.choices[0].message.content)
                common_themes = result.get("common_themes", [])
            except Exception as e:
                print(f"Error identifying common themes: {str(e)}")
                
                response = openai_client.chat.completions.create(
                    model=COMPLETION_MODEL,
                    messages=[
                        {"role": "system", "content": "You identify common themes between related sermons."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.5,
                    max_tokens=500,
                    response_format={"type": "json_object"}
                )
                
                result = json.loads(response.choices[0].message.content)
                common_themes = result.get("common_themes", [])
            except Exception as e:
                print(f"Error identifying common themes: {str(e)}")
        
        # Create response
        response_data = {
            "video_id": video_id,
            "sermon_title": enhanced_metadata.get("title", f"Sermon {video_id}"),
            "related_sermons": related_list,
            "total_related": len(related_list)
        }
        
        # Add common themes if available
        if common_themes:
            response_data["common_themes"] = common_themes
            
        # Cache the response
        if ENABLE_RESPONSE_CACHING:
            response_cache.set(cache_key, cache_params, response_data)
            
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error finding related sermons: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)