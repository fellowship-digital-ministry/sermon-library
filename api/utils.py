import os
import time
import glob
import json
import re
import calendar
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple

from fastapi import HTTPException, Header
from pydantic import BaseModel, Field
import openai
from pinecone import Pinecone

# Configure API keys and settings directly from environment variables
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT", "us-east-1")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "sermon-embeddings")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
SEARCH_TOP_K = int(os.environ.get("SEARCH_TOP_K", "5"))
COMPLETION_MODEL = os.environ.get("COMPLETION_MODEL", "gpt-4o")
TRANSLATION_MODEL = os.environ.get("TRANSLATION_MODEL", "gpt-4o")  # Smaller model for translations

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

# ============================
#        Data Models
# ============================
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

# ============================
#       Helper Functions
# ============================

def get_language_name(lang_code: str) -> str:
    """Return the full language name from a language code."""
    language_names = {
        "en": "English",
        "es": "Spanish",
        "zh": "Chinese",
    }
    return language_names.get(lang_code, "Unknown")

async def translate_text(text: str, source_lang: str, target_lang: str = "en"):
    """Translate text using GPT model."""
    if not text or source_lang == target_lang:
        return text  # No need to translate if languages match
    try:
        if target_lang == "en":
            prompt = f"Translate the following text from {get_language_name(source_lang)} to English. Preserve all information accurately: \n\n{text}"
        else:
            prompt = f"Translate the following text from English to {get_language_name(target_lang)}. Preserve all information accurately: \n\n{text}"
        response = openai_client.chat.completions.create(
            model=TRANSLATION_MODEL,
            messages=[{"role": "system", "content": "You are a professional translator."}, {"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Translation error: {str(e)}")
        return text

def get_language(accept_language: Optional[str] = Header(None, include_in_schema=False)) -> str:
    """Extract and validate the preferred language from the Accept-Language header."""
    if not accept_language:
        return "en"
    languages = [lang.split(";")[0].strip() for lang in accept_language.split(",")]
    for lang in languages:
        if lang.startswith("es"):
            return "es"
        elif lang.startswith("zh"):
            return "zh"
    return "en"

def load_metadata(video_id: str) -> Dict[str, Any]:
    """Load metadata for a video from its JSON file."""
    try:
        metadata_file = os.path.join(METADATA_DIR, f"{video_id}_metadata.json")
        with open(metadata_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"video_id": video_id, "title": f"Unknown Sermon ({video_id})", "publish_date": None}
    except json.JSONDecodeError:
        return {"video_id": video_id, "title": f"Error Loading Metadata ({video_id})", "publish_date": None}

