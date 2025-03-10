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


def generate_music_continuation(input_wav, model_name='facebook/musicgen-melody', progress=True):
    """
    Generates a continuation of the given input lofi audio file.

    Parameters:
    - input_wav (str): Path to the input .wav file (must be at least 20 seconds long).
    - model_name (str): Pretrained MusicGen model to use.
    - progress (bool): Whether to show generation progress.

    Returns:
    - Tensor: Generated audio waveform.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load the pretrained MusicGen model
    model = MusicGen.get_pretrained(model_name)

    # Load the input lofi audio
    prompt_waveform, prompt_sr = torchaudio.load(input_wav)
    num_samples = prompt_waveform.shape[1]

    # Ensure audio is at least 20 seconds long
    last_20_sec_samples = int(20 * prompt_sr)
    if num_samples < last_20_sec_samples:
        raise ValueError(f"Input audio must be at least 20 seconds long, but got {num_samples / prompt_sr:.2f} seconds.")

    # Trim to the last 20 seconds
    prompt_waveform = prompt_waveform[:, -last_20_sec_samples:]

    # Generate continuation with lofi mood in mind
    output = model.generate_continuation(prompt_waveform, prompt_sample_rate=prompt_sr, progress=progress)

    # Display and return the generated lofi track
    display_audio(output, sample_rate=32000)
    return output