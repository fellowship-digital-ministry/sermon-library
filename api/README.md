# Sermon Search API

A FastAPI backend for the sermon library that connects to Pinecone for vector search and OpenAI for generating answers.

## Overview

This API provides several endpoints for searching and querying sermon content:

- `/search` - Search for sermon segments matching a query
- `/answer` - Generate AI answers to questions based on sermon content
- `/sermons` - List all available sermons in the library
- `/sermons/{video_id}` - Get information about a specific sermon
- `/metadata` - Get statistics about metadata coverage

## Architecture

The API uses a two-tier architecture:

1. **Pinecone Vector Database** - Stores embeddings of sermon transcripts for semantic search
2. **Metadata JSON Files** - Provides rich metadata about each sermon

This design allows for efficient semantic search while maintaining detailed metadata without requiring revectorization.

## Metadata Integration

The API integrates metadata from JSON files in the following ways:

1. **Runtime Integration** - Automatically enriches search results with metadata from JSON files
2. **Periodic Synchronization** - Updates Pinecone vectors with metadata from JSON files through GitHub Actions

## Key Components

- `app.py` - Main FastAPI application with all endpoints
- `metadata_utils.py` - Utilities for accessing metadata from JSON files
- `improved_update_pinecone_metadata.py` - Script for updating Pinecone metadata from JSON files

## Environment Variables

The API requires the following environment variables:

- `OPENAI_API_KEY` - OpenAI API key
- `PINECONE_API_KEY` - Pinecone API key
- `PINECONE_ENVIRONMENT` - Pinecone environment (defaults to "us-east-1")
- `PINECONE_INDEX_NAME` - Pinecone index name (defaults to "sermon-embeddings")
- `EMBEDDING_MODEL` - OpenAI embedding model (defaults to "text-embedding-3-small")
- `COMPLETION_MODEL` - OpenAI completion model (defaults to "gpt-4o")

## Deployment

The API is deployed on Render.com and automatically updates when changes are pushed to the GitHub repository.

### Deployment URL

https://sermon-search-api-8fok.onrender.com

## Automation

The project includes GitHub Actions workflows for:

1. Monitoring and transcribing new sermons
2. Generating embeddings for new transcripts
3. Updating Pinecone metadata from JSON files

This ensures that both the transcript database and metadata are kept in sync.

## Manual Operations

### Initial Metadata Sync

To sync all metadata to Pinecone (one-time operation):

```bash
python api/improved_update_pinecone_metadata.py
```

### Update Recent Metadata

To update only recently changed metadata:

```bash
python api/improved_update_pinecone_metadata.py --only-recent --days=7
```

### Specific Sermon Updates

To update specific sermons by video ID:

```bash
python api/improved_update_pinecone_metadata.py --video-ids="video_id1,video_id2"
```