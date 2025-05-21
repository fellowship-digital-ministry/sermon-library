from fastapi import APIRouter, HTTPException, Query, Header

from .utils import (
    openai_client,
    pinecone_index,
    EMBEDDING_MODEL,
    COMPLETION_MODEL,
    SEARCH_TOP_K,
    SUBTITLES_DIR,
    METADATA_DIR,
    get_language,
    translate_text,
    load_metadata,
    SearchResponse,
    SearchResult,
    AnswerRequest,
    AnswerResponse,
    BibleReferenceStats,
)

from .search import (
    preprocess_query,
    extract_date_reference,
    format_date_for_human,
    get_proper_date_from_metadata,
    format_response_with_suggestions,
    generate_ai_answer_with_suggestions,
    generate_suggested_queries_with_results,
    generate_suggested_queries,
    is_off_topic_query,
    generate_no_results_message,
    generate_embedding,
    format_time,
    get_youtube_timestamp_url,
    generate_enhanced_ai_answer,
)
from .bible import load_bible_references, get_bible_stats

router = APIRouter()
# API Endpoints
@router.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Sermon Search API is running",
        "documentation": "/docs",
        "version": "1.1.0"  # Updated version number
    }

@router.get("/health")
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

@router.get("/search", response_model=SearchResponse)
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

@router.post("/answer", response_model=AnswerResponse)
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
    
@router.get("/transcript/{video_id}")
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

@router.get("/sermons")
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

@router.get("/sermons/{video_id}")
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
@router.get("/bible/stats", response_model=BibleReferenceStats)
async def get_bible_reference_stats():
    """Get statistics about Bible references in sermons."""
    try:
        stats = get_bible_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating Bible stats: {str(e)}")

@router.get("/bible/books")
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

@router.get("/bible/books/{book}")
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

@router.get("/bible/references/{reference_id}")
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


