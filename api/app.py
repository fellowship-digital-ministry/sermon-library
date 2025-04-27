"""
Sermon Search API

A FastAPI backend for the sermon library that connects to Pinecone
for vector search and OpenAI for generating answers.
"""

import os
import time
from typing import List, Dict, Optional, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
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

# Check for required environment variables
if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY environment variable")
if not PINECONE_API_KEY:
    raise ValueError("Missing PINECONE_API_KEY environment variable")

# Initialize clients
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Initialize Pinecone with new API
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
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
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

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total_results: int
    processing_time: float

class AnswerRequest(BaseModel):
    query: str = Field(..., description="The question to answer based on sermon content")
    top_k: int = Field(5, description="Number of search results to consider")
    include_sources: bool = Field(True, description="Whether to include source information in the response")

class AnswerResponse(BaseModel):
    query: str
    answer: str
    sources: List[SearchResult] = []
    processing_time: float

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

def generate_ai_answer(query: str, search_results: List[SearchResult]) -> str:
    """Generate an AI answer based on the search results."""
    # Prepare the context from search results
    context = "\n\n".join([
        f"Segment {i+1} (Time: {format_time(result.start_time)} - {format_time(result.end_time)}):\n{result.text}"
        for i, result in enumerate(search_results)
    ])
    
    # Prepare the prompt for GPT-4
    prompt = f"""
You are an assistant helping users understand sermon content. Answer the following question based only on the 
provided sermon segments. If the answer cannot be found in the segments, say so clearly.

USER QUESTION: {query}

SERMON SEGMENTS:
{context}

Answer the question based only on the provided sermon segments. Include specific references to which segment(s) 
contain the information (e.g., "In Segment 3, the pastor explains..."). Keep your response focused and concise.
    """
    
    try:
        response = openai_client.chat.completions.create(
            model=COMPLETION_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions about sermon content."},
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
        # Check Pinecone
        stats = pinecone_index.describe_index_stats()
        vector_count = stats.get("total_vector_count", 0)
        
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
        
        # Search Pinecone
        search_response = pinecone_index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        # Format results
        results = []
        matches = search_response.get("matches", [])
        
        for match in matches:
            if match["score"] < min_score:
                continue
                
            metadata = match["metadata"]
            
            # Convert segment_ids to List[str] if needed
            segment_ids = metadata.get("segment_ids", [])
            if not isinstance(segment_ids, list):
                segment_ids = []
            
            results.append(SearchResult(
                video_id=metadata.get("video_id", ""),
                title=metadata.get("title", "Unknown Sermon"),
                url=get_youtube_timestamp_url(metadata.get("video_id", ""), metadata.get("start_time", 0)),
                text=metadata.get("text", ""),
                start_time=metadata.get("start_time", 0),
                end_time=metadata.get("end_time", 0),
                similarity=match["score"],
                chunk_index=metadata.get("chunk_index", 0),
                segment_ids=segment_ids
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
        
        # Search Pinecone
        search_response = pinecone_index.query(
            vector=query_embedding,
            top_k=request.top_k,
            include_metadata=True
        )
        
        # Format search results
        search_results = []
        matches = search_response.get("matches", [])
        
        for match in matches:
            if match["score"] < 0.5:  # Minimum threshold for relevance
                continue
                
            metadata = match["metadata"]
            
            # Convert segment_ids to List[str] if needed
            segment_ids = metadata.get("segment_ids", [])
            if not isinstance(segment_ids, list):
                segment_ids = []
            
            search_results.append(SearchResult(
                video_id=metadata.get("video_id", ""),
                title=metadata.get("title", "Unknown Sermon"),
                url=get_youtube_timestamp_url(metadata.get("video_id", ""), metadata.get("start_time", 0)),
                text=metadata.get("text", ""),
                start_time=metadata.get("start_time", 0),
                end_time=metadata.get("end_time", 0),
                similarity=match["score"],
                chunk_index=metadata.get("chunk_index", 0),
                segment_ids=segment_ids
            ))
        
        # Generate AI answer
        answer_text = "No relevant sermon content found to answer this question."
        if search_results:
            answer_text = generate_ai_answer(request.query, search_results)
        
        return AnswerResponse(
            query=request.query,
            answer=answer_text,
            sources=search_results if request.include_sources else [],
            processing_time=time.time() - start_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Answer generation error: {str(e)}")

@app.get("/sermons")
async def list_sermons(
    limit: int = Query(50, description="Maximum number of sermons to return"),
    offset: int = Query(0, description="Offset for pagination")
):
    """
    List available sermons in the library.
    Returns metadata for available sermons.
    """
    try:
        # Get stats from Pinecone
        stats = pinecone_index.describe_index_stats()
        
        # Query for a sample of vectors to get sermon metadata
        results = pinecone_index.query(
            vector=[0.1] * 1536,  # Random vector
            top_k=limit + offset,
            include_metadata=True
        )
        
        # Group by video_id to get unique sermons
        sermons = {}
        matches = results.get("matches", [])
        
        for match in matches:
            metadata = match["metadata"]
            video_id = metadata.get("video_id", "")
            
            if video_id and video_id not in sermons:
                sermons[video_id] = {
                    "video_id": video_id,
                    "title": metadata.get("title", f"Sermon {video_id}"),
                    "channel": metadata.get("channel", "Unknown"),
                    "publish_date": metadata.get("publish_date", ""),
                    "url": f"https://www.youtube.com/watch?v={video_id}"
                }
        
        # Convert to list and apply pagination
        sermon_list = list(sermons.values())
        sermon_list.sort(key=lambda x: x.get("publish_date", ""), reverse=True)
        
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
        # Query Pinecone for chunks with this video_id
        # Use a filter to get only chunks for this sermon
        query_embedding = generate_embedding("sermon about faith")  # Generic query
        
        # Create a filter to match video_id
        filter_dict = {"video_id": {"$eq": video_id}}
        
        results = pinecone_index.query(
            vector=query_embedding,
            top_k=100,  # Get many results to find chunks from this sermon
            include_metadata=True,
            filter=filter_dict
        )
        
        matches = results.get("matches", [])
        if not matches:
            raise HTTPException(status_code=404, detail=f"Sermon not found: {video_id}")
        
        # Get sermon metadata from the first match
        first_match = matches[0]["metadata"]
        sermon_info = {
            "video_id": video_id,
            "title": first_match.get("title", f"Sermon {video_id}"),
            "channel": first_match.get("channel", "Unknown"),
            "publish_date": first_match.get("publish_date", ""),
            "url": f"https://www.youtube.com/watch?v={video_id}"
        }
        
        # Collect all chunks
        chunks = []
        for match in matches:
            metadata = match["metadata"]
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