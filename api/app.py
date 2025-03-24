from flask import Flask, request, jsonify
import sys
import os
import threading
import time

# Add the parent directory of 'ml_scripts' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml_scripts.generate_textinput import generate_long_music, save_audio

app = Flask(__name__)

# Dictionary to store progress for each song generation.
# For progress, we'll use:
#  - 0 to 100 for percentage progress,
#  - -1 to indicate an error.
song_progress = {}

def background_generate_music(song_id, prompt):
    """
    Function to generate music in the background.
    It simulates progress updates and calls your actual generation function.
    """
    try:
        song_progress[song_id] = 0
        
        # Simulate progress updates in 5 steps (20% increments)
        for i in range(1, 6):
            time.sleep(2)  # simulate a step in the generation process
            song_progress[song_id] = i * 20  # update progress

        # Generate the music (this may also take some time)
        audio = generate_long_music(prompt)
        sample_rate = 44100  # Adjust based on your model's output
        
        # Save the generated audio with a unique file name using the song_id
        save_audio(audio, sample_rate, f"generated_music_{song_id}.wav")
        
        # Mark the process as complete
        song_progress[song_id] = 100
    except Exception as e:
        # Mark with -1 on error, optionally you could log or store the exception
        song_progress[song_id] = -1

@app.route('/generate_music', methods=['POST'])
def generate_music_endpoint():
    data = request.get_json()
    prompt = data.get('prompt', '')

    if not prompt:
        return jsonify({'error': 'Prompt is required'}), 400

    # Generate a timestamp-based unique song id
    timestamp = int(time.time() * 1000)  # current time in milliseconds
    song_id = f"song_{timestamp}"

    # Start the background song generation in a separate thread
    thread = threading.Thread(target=background_generate_music, args=(song_id, prompt))
    thread.start()

    # Return the song id immediately and initial progress (0)
    return jsonify({'song_id': song_id, 'progress': song_progress.get(song_id, 0)}), 202

@app.route('/progress/<song_id>', methods=['GET'])
def get_progress(song_id):
    progress = song_progress.get(song_id)
    if progress is None:
        return jsonify({'error': 'Invalid song_id or song not found'}), 404
    elif progress == -1:
        return jsonify({'song_id': song_id, 'error': 'Song generation failed'}), 500
    else:
        return jsonify({'song_id': song_id, 'progress': progress}), 200

if __name__ == '__main__':
    app.run(debug=True)