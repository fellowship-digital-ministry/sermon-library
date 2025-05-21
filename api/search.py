import os
import glob
import json
import re
import calendar
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any

from fastapi import HTTPException

from .utils import (
    METADATA_DIR,
    openai_client,
    pinecone_index,
    EMBEDDING_MODEL,
    COMPLETION_MODEL,
    SearchResult,
    load_metadata,
)

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


