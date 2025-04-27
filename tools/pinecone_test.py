"""
Test script to verify Pinecone connection with direct API key input
"""

from pinecone import Pinecone

# Replace this with your actual API key when running the test
API_KEY = "pcsk_khcUz_RPtxQCVimQASNLzn6qDhsXMxBDEEVgh7fHHJFAUqtXVbEEu18gjXjpVBDyUR2Vi"  
ENVIRONMENT = "us-east-1"
INDEX_NAME = "sermon-embeddings"

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