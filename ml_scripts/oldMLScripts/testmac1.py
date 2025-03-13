import torch
import torchaudio
from audiocraft.models import MusicGen

# 1. Device selection: Use MPS if available, otherwise CPU
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")  # Debugging output to show selected device

# 1. Load the pre-trained MusicGen model. If using MPS, load on CPU first to avoid any MPS-specific loading issues.
model = MusicGen.get_pretrained('facebook/musicgen-melody', device='cpu')

if device.type == 'mps':
    # Move the heavy language model (LM) to MPS for faster generation, keep the audio compression model on CPU.
    model.lm.to(device)
    model.device = device  # Update model's device attribute to MPS
    # 4. Disable autocast for MPS to avoid unsupported operations (MPS autocast can cause errors).
    class DummyAutocast:
        def __enter__(self): return None  # no-op context
        def __exit__(self, exc_type, exc_val, exc_tb): return False
    model.autocast = DummyAutocast()

    # 2. Patch the decode method to run on CPU (prevents garbled audio on MPS decode [oai_citation_attribution:0‡github.com](https://github.com/facebookresearch/audiocraft/issues/31#:~:text=,decoder%28emb)).
    original_decode = model.compression_model.decode
    def decode_on_cpu(codes: torch.Tensor, scale: torch.Tensor = None):
        # Decode on CPU to avoid MPS decoder issues
        codes_cpu = codes.to("cpu")
        with torch.no_grad():
            audio = original_decode(codes_cpu, scale)  # decode with CPU weights
        return audio.to(device)  # move result back to MPS (keep tensor on MPS if model expects it)
    model.compression_model.decode = decode_on_cpu

    # 3. Patch the encode method to run on CPU (avoids MPS encoder issues [oai_citation_attribution:1‡github.com](https://github.com/facebookresearch/audiocraft/issues/31#:~:text=x%2C%20scale%20%3D%20self.preprocess%28x%29%20,else) for melody or prompt audio).
    original_encode = model.compression_model.encode
    def encode_on_cpu(x: torch.Tensor):
        # Encode on CPU to avoid MPS encoder issues
        x_cpu = x.to("cpu")
        with torch.no_grad():
            codes, scale = original_encode(x_cpu)  # encode with CPU weights
        # Move encoded tokens back to MPS for further processing by the LM
        return codes.to(device), (scale.to(device) if scale is not None else None)
    model.compression_model.encode = encode_on_cpu

# 5. Device transfers are handled above (CPU↔MPS for encode/decode). Now set generation params (e.g., 8 seconds duration).
model.set_generation_params(duration=8)

# Prepare inputs for generation
descriptions = ["Calm lofi music with a happy tone."]  # text prompt
# If using melody conditioning, load the melody (keep it on CPU to use CPU encoding):
melody_waveform, sr = torchaudio.load('./assets/lofi1.mp3')
# (Do not move melody_waveform to MPS. The patched encode will handle moving it to CPU if needed.)

# Generate music using the text and optional melody conditioning
if 'melody' in model.name:
    # Melody model: condition on melody audio
    output_waveforms = model.generate_with_chroma(descriptions, melody_waveform, sr)
else:
    # Text-only model
    output_waveforms = model.generate(descriptions)

# Move output to CPU for saving or playback
output_waveforms = output_waveforms.cpu().detach()
torchaudio.save('output.wav', output_waveforms[0], sample_rate=model.sample_rate)