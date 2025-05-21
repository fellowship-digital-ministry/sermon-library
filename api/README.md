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

The API integrates metadata from JSON files at runtime so search results include sermon details.

## Key Components

- `app.py` - Main FastAPI application with all endpoints
- `metadata_utils.py` - Utilities for accessing metadata from JSON files

## Environment Variables

The API requires the following environment variables:

- `OPENAI_API_KEY` - OpenAI API key
- `PINECONE_API_KEY` - Pinecone API key
- `PINECONE_ENVIRONMENT` - Pinecone environment (defaults to "us-east-1")
- `PINECONE_INDEX_NAME` - Pinecone index name (defaults to "sermon-embeddings")
- `EMBEDDING_MODEL` - OpenAI embedding model (defaults to "text-embedding-3-small")
- `COMPLETION_MODEL` - OpenAI completion model (defaults to "gpt-4o")

## Running Locally

To run the API on your machine you must provide the required environment
variables (e.g. `OPENAI_API_KEY` and `PINECONE_API_KEY`). Once they are set,
start the server with `uvicorn`:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Launch the command from the `api` directory so FastAPI can locate `app.py`.

## Deployment

The API is deployed on Render.com and automatically updates when changes are pushed to the GitHub repository.

### Deployment URL

https://sermon-search-api-8fok.onrender.com

## Automation

The project includes GitHub Actions workflows for:

1. Monitoring and transcribing new sermons
2. Generating embeddings for new transcripts

This ensures that both the transcript database and metadata are kept in sync.
