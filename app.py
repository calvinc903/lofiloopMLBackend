from flask import Flask, request, jsonify
import time
import threading

from ml_scripts.generate_textinput import generate_long_music, save_audio, load_model
from api.aws import upload_file_to_s3, BUCKET_NAME
from api.mongo_util import insert_song_metadata

app = Flask(__name__)

# ✅ Global shared variables
song_progress = {}
generation_in_progress = False

# ✅ Pre-load model at startup
model = load_model()

def background_generate(song_id, prompt):
    global generation_in_progress

    try:
        print(f"[INFO] Starting generation for {song_id}...")

        # Generate music
        audio = generate_long_music(prompt, song_id, model=model, song_progress=song_progress)
        sample_rate = model.sample_rate
        filename = f"generated_music_{song_id}.wav"
        save_audio(audio, sample_rate, filename)

        # Upload to AWS S3
        cloudfront_url, object_key = upload_file_to_s3(filename, BUCKET_NAME)

        # Store metadata in MongoDB
        metadata = {
            "song_id": song_id,
            "aws_url": cloudfront_url,
            "object_key": object_key,
            "progress": 100,
            "prompt": prompt,
        }
        insert_song_metadata(metadata)

        # Update progress tracking
        song_progress[song_id].update({"progress": 100, "aws_url": cloudfront_url})
        print(f"[SUCCESS] Generation completed for {song_id}.")
        print(f"[INFO] CloudFront URL for {song_id}: {cloudfront_url}")

    except Exception as e:
        song_progress[song_id] = {"error": str(e), "progress": -1}
        print(f"[ERROR] Generation failed for {song_id}: {e}")

    finally:
        generation_in_progress = False  # ✅ Always clear lock at end

@app.route('/generate_music', methods=['POST'])
def generate_music_endpoint():
    global generation_in_progress

    if generation_in_progress:
        return jsonify({'error': 'Generation already in progress'}), 429

    data = request.get_json()
    prompt = data.get('prompt', '')
    if not prompt:
        return jsonify({'error': 'Prompt is required'}), 400

    # Generate unique song ID
    timestamp = int(time.time() * 1000)
    song_id = f"song_{timestamp}"
    song_progress[song_id] = {"progress": 0, "start_time": time.time()}

    # ✅ Set lock and start background thread
    generation_in_progress = True
    thread = threading.Thread(target=background_generate, args=(song_id, prompt))
    thread.start()

    print(f"[INFO] Accepted generation request for {song_id}. Responding immediately.")

    # ✅ Return immediately with song ID
    return jsonify({'song_id': song_id}), 202

@app.route('/progress/<song_id>', methods=['GET'])
def get_progress(song_id):
    progress = song_progress.get(song_id)
    if progress is None:
        return jsonify({'error': 'Invalid song_id or song not found'}), 404
    elif isinstance(progress, dict) and progress.get('progress') == -1:
        return jsonify({
            'song_id': song_id,
            'error': 'Generation failed',
            'details': progress.get('error')
        }), 500
    else:
        elapsed = None
        if 'start_time' in progress:
            elapsed = time.time() - progress['start_time']

        # ✅ Optionally surface CloudFront URL directly for convenience
        response = {
            'song_id': song_id,
            'progress': progress,
            'elapsed_seconds': round(elapsed, 2) if elapsed is not None else None
        }
        if 'aws_url' in progress:
            response['download_url'] = progress['aws_url']

        return jsonify(response), 200

@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        'status': 'running',
        'generation_in_progress': generation_in_progress
    }), 200

if __name__ == '__main__':
    app.run(debug=True)