import torchaudio
import torch
import time
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from audiocraft.utils.notebook import display_audio

def generate_music_from_text(description, output_path="output", duration=30):
    """
    Generates music based only on a text description.

    Parameters:
    - description (str): The text description of the music.
    - output_path (str): Path to save the generated output.
    - duration (int): Duration of generated audio in seconds.

    Returns:
    - str: Path to the saved generated audio file.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = MusicGen.get_pretrained('facebook/musicgen-melody')
    model.set_generation_params(duration=duration)

    start_time = time.time()

    wav = model.generate([description])[0]  # Generate from text only

    end_time = time.time()
    print(f"Track generation took {end_time - start_time:.2f} seconds.")

    output_file = f"{output_path}"
    audio_write(output_file, wav[0].cpu(), model.sample_rate, strategy="loudness")

    print(f"Generated track saved at: {output_file}")
    return output_file


def generate_music_continuation(input_audio, output_path="generated_music.wav", duration=10, model_name="facebook/musicgen-melody"):
    """
    Generates a continuation of an input music segment.

    Parameters:
    - input_audio (str): Path to the input audio file (.wav, .mp3, etc.).
    - output_path (str): Path to save the generated output (default: "generated_music.wav").
    - duration (int): Duration of generated continuation in seconds.
    - model_name (str): The MusicGen model to use (default: "facebook/musicgen-melody").

    Returns:
    - str: Path to the saved generated audio file.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load pretrained MusicGen model
    model = MusicGen.get_pretrained(model_name)
    model.set_generation_params(duration=duration)

    # Load input audio
    prompt_waveform, prompt_sr = torchaudio.load(input_audio)
    
    # Ensure the prompt isn't too long
    max_prompt_duration = 10  # MusicGen expects short prompts (adjustable)
    required_samples = int(max_prompt_duration * prompt_sr)
    prompt_waveform = prompt_waveform[:, -required_samples:]  # Trim to last 10 seconds

    # Generate continuation
    output_waveform = model.generate_continuation(prompt_waveform, prompt_sample_rate=prompt_sr, progress=True)

    # Ensure output is a 2D tensor before saving
    print(f"Generated output shape: {output_waveform.shape}")  # Debugging
    if output_waveform.ndim == 3:  # Shape: [batch, channels, samples]
        output_waveform = output_waveform.squeeze(0)
    elif output_waveform.ndim != 2:
        raise ValueError(f"Unexpected tensor shape: {output_waveform.shape}")

    # Save generated audio
    sample_rate = getattr(model, "sample_rate", 32000)
    torchaudio.save(output_path, output_waveform.cpu().detach(), sample_rate)

    print(f"Generated continuation saved at: {output_path}")
    return output_path
