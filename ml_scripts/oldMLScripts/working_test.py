import torchaudio
import torch
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

# Set device to MPS if available, otherwise use CPU
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# Test GPU usage with a dummy tensor
x = torch.randn(1000, 1000).to(device)  # Move to MPS
y = x @ x  # Perform a matrix multiplication
print("GPU computation complete!")

# Load MusicGen model (MusicGen does not have .to(device), so we avoid that)
model = MusicGen.get_pretrained('facebook/musicgen-melody')
model.set_generation_params(duration=30)  # generate 8 seconds.

descriptions = ['chill lofi music']

# Load the melody and move it to MPS
melody, sr = torchaudio.load('./assets/lofi1.mp3')
melody = melody.to(device)  # Move melody tensor to GPU

# Generates using the melody and single description
wav = model.generate_with_chroma(descriptions, melody[None], sr)  # No need to expand

# Save the output
audio_write('output', wav[0].cpu(), model.sample_rate, strategy="loudness")

print("Music generation complete!")