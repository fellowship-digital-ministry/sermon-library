"""
Test script to verify Pinecone connection using environment variables.

Set the following environment variables before running:
 - ``PINECONE_API_KEY``
 - ``PINECONE_ENVIRONMENT``
 - ``PINECONE_INDEX_NAME``
"""

from pinecone import Pinecone
import os
import sys

API_KEY = os.environ.get("PINECONE_API_KEY")
ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT")
INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME")

missing = [
    name for name, value in {
        "PINECONE_API_KEY": API_KEY,
        "PINECONE_ENVIRONMENT": ENVIRONMENT,
        "PINECONE_INDEX_NAME": INDEX_NAME,
    }.items()
    if not value
]

if missing:
    print(f"Error: missing environment variables: {', '.join(missing)}")
    sys.exit(1)

try:
    # Initialize Pinecone with direct API key
    print(f"Connecting to Pinecone...")
    pc = Pinecone(api_key=API_KEY)
    
    print("Successfully initialized Pinecone!")
    
    # List indexes
    indexes = pc.list_indexes()
    print(f"Available indexes: {indexes}")
    
    # Check if our index exists
    index_names = [index.name for index in indexes]
    print(f"Index names: {index_names}")
    
    if INDEX_NAME in index_names:
        print(f"Found index: {INDEX_NAME}")
        
        # Connect to index
        index = pc.Index(INDEX_NAME)
        
        # Get index stats
        stats = index.describe_index_stats()
        print(f"Index stats: {stats}")
    else:
        print(f"Index '{INDEX_NAME}' not found in available indexes")
    
except Exception as e:
    print(f"Error: {str(e)}")

print("Test complete!")