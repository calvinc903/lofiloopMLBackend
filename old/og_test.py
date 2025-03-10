import torchaudio
import torch
import time
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the pretrained model (DO NOT call .to(device))
model = MusicGen.get_pretrained('facebook/musicgen-melody')
model.set_generation_params(duration=30)  # Generate 30 seconds.

# Use a single description.
description = 'chill happy lofi music'

# Load the melody and move it to the correct device
melody, sr = torchaudio.load('./assets/lofi1.mp3')
melody = melody.to(device)

# Start timing
start_time = time.time()

# Generate using the melody and description
wav = model.generate_with_chroma([description], melody[None], sr)

# End timing
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Track generation took {elapsed_time:.2f} seconds.")

# Save the generated audio clip
audio_write('output', wav[0].cpu(), model.sample_rate, strategy="loudness")