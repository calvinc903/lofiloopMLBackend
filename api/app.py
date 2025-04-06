from flask import Flask, request, jsonify
import sys
import os
import threading
import time

# Add the parent directory of 'ml_scripts' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml_scripts.generate_textinput import generate_long_music, save_audio, song_progress
from aws import upload_file_to_s3, BUCKET_NAME  # Your AWS upload code
from mongo_util import insert_song_metadata  # Import the utility function

app = Flask(__name__)

def background_generate_music(song_id, prompt):
    def run_generation():
        try:
            # Generate and save the song locally.
            audio = generate_long_music(prompt, song_id)
            sample_rate = 44100  
            filename = f"generated_music_{song_id}.wav"
            save_audio(audio, sample_rate, filename)
            
            # Upload the song file to AWS S3.
            cloudfront_url, object_key = upload_file_to_s3(filename, BUCKET_NAME)
            
            # Prepare metadata to store in MongoDB.
            metadata = {
                "song_id": song_id,
                "aws_url": cloudfront_url,
                "object_key": object_key,
                "progress": 100,
                "prompt": prompt,
            }
            # Insert metadata into MongoDB.
            insert_song_metadata(metadata)
            
            # Update the progress store while preserving the start_time.
            if song_id in song_progress and isinstance(song_progress[song_id], dict) and "start_time" in song_progress[song_id]:
                song_progress[song_id].update({"progress": 100, "aws_url": cloudfront_url})
            else:
                song_progress[song_id] = {"progress": 100, "aws_url": cloudfront_url}
        except Exception as e:
            song_progress[song_id] = {"error": str(e), "progress": -1}
    
    thread = threading.Thread(target=run_generation)
    thread.start()

@app.route('/generate_music', methods=['POST'])
def generate_music_endpoint():
    data = request.get_json()
    prompt = data.get('prompt', '')
    if not prompt:
        return jsonify({'error': 'Prompt is required'}), 400

    timestamp = int(time.time() * 1000) 
    song_id = f"song_{timestamp}"
    
    # Record the start time in the progress dictionary.
    song_progress[song_id] = {"progress": 0, "start_time": time.time()}
    
    # Start the background generation process.
    background_generate_music(song_id, prompt)
    
    # Return the song_id immediately.
    return jsonify({'song_id': song_id, 'progress': song_progress.get(song_id, 0)}), 202

@app.route('/progress/<song_id>', methods=['GET'])
def get_progress(song_id):
    progress = song_progress.get(song_id)
    if progress is None:
        return jsonify({'error': 'Invalid song_id or song not found'}), 404
    elif isinstance(progress, dict) and progress.get('progress') == -1:
        return jsonify({'song_id': song_id, 'error': 'Song generation failed', 'details': progress.get('error')}), 500
    else:
        elapsed = None
        if 'start_time' in progress:
            elapsed = time.time() - progress['start_time']
        return jsonify({
            'song_id': song_id,
            'progress': progress,
            'elapsed_seconds': round(elapsed, 2) if elapsed is not None else None
        }), 200

if __name__ == '__main__':
    app.run(debug=True)