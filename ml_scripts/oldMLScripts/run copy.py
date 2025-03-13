from generate_textandaudioinput import generate_music_from_text, generate_music_continuation, generate_long_music
from old.trackModification import concatenate_tracks
import os
import torchaudio
import time


# # Step 1: Generate a track from text description
# description = "Lofi slow bpm electro chill with organic samples"
# first_track = generate_music_from_text(description, output_path="text_generated_track", duration=30)

# time.sleep(2)

# # Step 2: Use the first generated track as input to generate a new track
# first_track += ".wav"
# second_track = generate_music_continuation("text_generated_track.wav", output_path="continued_music", duration=30)

# second_track += ".wav"
# # Step 3: Combine both tracks into a final result
# final_track = concatenate_tracks([first_track, second_track], output_path="final_combined_lofi")

# print(f"Final combined track saved at: {final_track}")


prompt = "Lofi slow bpm electro chill with organic samples."
prompt = "ambient sound with minimal fluctuation in the melody that also has brown noise into it. make it sound like you're in a silent private jet.."

waveform, sample_rate = generate_long_music(prompt)
torchaudio.save("long_generated_music.wav", waveform.cpu(), sample_rate)