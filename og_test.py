import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

# Load the pretrained model and set generation duration to 30 seconds.
model = MusicGen.get_pretrained('melody')
model.set_generation_params(duration=30)  # generate 30 seconds.

# Use a single description.
description = 'chill happy lofi music'

melody, sr = torchaudio.load('./assets/lofi1.mp3')
# Generate using the melody from the provided audio and the description.
# Note: unsqueeze once to have a batch dimension.
wav = model.generate_with_chroma([description], melody[None], sr)

# Save the generated audio clip.
audio_write('output', wav[0].cpu(), model.sample_rate, strategy="loudness")