import boto3
import time

# Configuration (Update with your values)
BUCKET_NAME = "lofi-loop-song-storage"
CLOUDFRONT_DOMAIN = "https://d3sadghi8jltyo.cloudfront.net"
LOCAL_FILE_PATH = "generated_music.wav"  # Local file to be uploaded

# Initialize S3 client
s3_client = boto3.client("s3")

def upload_file_to_s3(local_file, bucket):
    """Uploads a file to S3 with a timestamp-based unique key and returns its CloudFront URL."""
    try:
        # Generate a unique object key using the current time in milliseconds
        timestamp = int(time.time() * 1000)
        object_key = f"song_{timestamp}.wav"
        
        # Upload file to S3 with the unique key
        s3_client.upload_file(local_file, bucket, object_key, ExtraArgs={'ContentType': 'audio/wav'})
        print(f"File uploaded successfully: {object_key}")

        # Generate CloudFront URL using the unique key
        cloudfront_url = f"{CLOUDFRONT_DOMAIN}/{object_key}"
        return cloudfront_url, object_key
    except Exception as e:
        print(f"Error uploading file: {e}")
        return None, None

# Upload file and get CloudFront URL and unique S3 object key
cloudfront_url, object_key = upload_file_to_s3(LOCAL_FILE_PATH, BUCKET_NAME)

if cloudfront_url:
    print(f"Access your file at: {cloudfront_url}")