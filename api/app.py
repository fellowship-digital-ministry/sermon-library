import os
import time
from typing import List, Dict, Optional, Any, Union
from datetime import datetime
import json

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

# Path to metadata directory
METADATA_DIR = os.environ.get("METADATA_DIR", "./transcription/data/metadata")
SUBTITLES_DIR = os.environ.get("SUBTITLES_DIR", "./transcription/data/subtitles")

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

class AnswerResponse(BaseModel):
    query: str
    answer: str
    sources: List[SearchResult] = []
    processing_time: float

# Helper Functions
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
    return f"{minutes:02d}:{seconds:02d}"

def get_youtube_timestamp_url(video_id: str, seconds: float) -> str:
    """Generate a YouTube URL with a timestamp."""
    return f"https://www.youtube.com/watch?v={video_id}&t={int(seconds)}"

def generate_ai_answer(query: str, search_results: List[SearchResult], language: str = "en") -> str:
    """Generate an AI answer based on the search results in the specified language."""
    # Prepare the context from search results
    context = "\n\n".join([
        f"Segment {i+1} (Time: {format_time(result.start_time)} - {format_time(result.end_time)}):\n{result.text}"
        for i, result in enumerate(search_results)
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

Responde a la pregunta basándote únicamente en los segmentos de sermón proporcionados. Incluye referencias específicas a qué segmento(s) contienen la información (por ejemplo, "En el Segmento 3, el pastor explica..."). Mantén tu respuesta enfocada y concisa.
        """
    elif language == "zh":
        prompt = f"""
根据提供的讲道片段回答以下问题。如果在这些片段中找不到答案，请清楚地说明。

用户问题: {query}

讲道片段:
{context}

仅根据提供的讲道片段回答问题。包括具体引用哪个片段包含信息（例如，"在片段3中，牧师解释了..."）。保持回答重点明确和简洁。
        """
    else:
        prompt = f"""
Answer the following question based only on the provided sermon segments. If the answer cannot be found in the segments, say so clearly.

USER QUESTION: {query}

SERMON SEGMENTS:
{context}

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
    """
    start_time = time.time()
    
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
                publish_date=enhanced_metadata.get("publish_date")  # Add this line
            ))
        
        return SearchResponse(
            query=query,
            results=results,
            total_results=len(results),
            processing_time=time.time() - start_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

@app.post("/answer", response_model=AnswerResponse)
async def answer(request: AnswerRequest):
    """
    Generate an AI answer to a question based on sermon content.
    Searches for relevant sermon segments and uses them to create a response.
    """
    start_time = time.time()
    
    try:
        # Generate embedding for the query
        query_embedding = generate_embedding(request.query)
        
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
                publish_date=enhanced_metadata.get("publish_date")  # Add this line
            ))
        
        # Generate AI answer - pass language to the generation function
        answer_text = "No relevant sermon content found to answer this question."
        if search_results:
            answer_text = generate_ai_answer(request.query, search_results, request.language)
        
        return AnswerResponse(
            query=request.query,
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
                
                segments.append({
                    "start_time": start_time,
                    "end_time": end_time,
                    "text": text
                })
            
            # If language isn't English, add a note about language availability
            note = ""
            if language != "en":
                note = "Transcripts are currently only available in English. Future updates will include translation."
            
            return {
                "video_id": video_id,
                "title": enhanced_metadata.get("title", f"Sermon {video_id}"),
                "publish_date": enhanced_metadata.get("publish_date"),
                "language": language,
                "segments": segments,
                "total_segments": len(segments),
                "note": note,
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
                raw_segments.append({
                    "start_time": metadata.get("start_time", 0),
                    "end_time": metadata.get("end_time", 0),
                    "text": metadata.get("text", ""),
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
        
        # If language isn't English, add a note about language availability
        note = ""
        if language != "en":
            note = "Transcripts are currently only available in English. Future updates will include translation."
        
        return {
            "video_id": video_id,
            "title": enhanced_metadata.get("title", f"Sermon {video_id}"),
            "publish_date": enhanced_metadata.get("publish_date"),
            "language": language,
            "segments": processed_segments,
            "total_segments": len(processed_segments),
            "note": note,
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)