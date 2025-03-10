from generate import generate_music_from_text, generate_music_from_audio, concatenate_tracks
import os
import time


# Step 1: Generate a track from text description
description = "Lofi slow bpm electro chill with organic samples"
first_track = generate_music_from_text(description, output_path="text_generated_track", duration=30)

time.sleep(2)

# Step 2: Use the first generated track as input to generate a new track
first_track += ".wav"
second_track = generate_music_from_audio(first_track, output_path="audio_generated_track", duration=30)

# Step 3: Combine both tracks into a final result
final_track = concatenate_tracks([first_track, second_track], output_path="final_combined_lofi")

print(f"Final combined track saved at: {final_track}")