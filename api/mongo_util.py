# mongo_util.py
from pymongo import MongoClient
import time

MONGO_URI = "mongodb+srv://admin:78d5T2Z7Gga4@lofiloopaccountsdb.q7bsu.mongodb.net/?retryWrites=true&w=majority&appName=LoFiLoopAccountsDB"
client = MongoClient(MONGO_URI)

# Use a database name (e.g., "lofiloop_db") and a collection name (e.g., "songs_metadata").
db = client['music_db']
collection = db['songs_metadata']

def insert_song_metadata(metadata: dict):
    """
    Insert song metadata into MongoDB.
    
    Expected metadata keys might include:
      - song_id: str
      - aws_url: str (CloudFront URL)
      - object_key: str (the key used in S3)
      - progress: int (final progress status, e.g., 100)
      - timestamp: float (time of insertion)
      - any other relevant fields
    """
    # Optionally, add a timestamp if not provided
    if "timestamp" not in metadata:
        metadata["timestamp"] = time.time()
    result = collection.insert_one(metadata)
    print("Inserted metadata with id:", result.inserted_id)
    return result.inserted_id

# Test the insertion when run directly.
if __name__ == "__main__":
    test_metadata = {
        "song_id": "song_123456789",
        "aws_url": "https://d3sadghi8jltyo.cloudfront.net/song_123456789.wav",
        "object_key": "song_123456789.wav",
        "progress": 100,
    }
    insert_song_metadata(test_metadata)