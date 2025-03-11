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


import torch
from audiocraft.models import MusicGen

def generate_long_music(prompt, total_duration=120, segment_duration=30, overlap_duration=20, model_name='facebook/musicgen-melody'):
    """
    Generates a long music track using a sliding window approach.

    Parameters:
    - prompt (str): Text prompt describing the desired music.
    - total_duration (int): Total length of the generated music in seconds.
    - segment_duration (int): Duration of each generated segment in seconds.
    - overlap_duration (int): Overlap between consecutive segments in seconds.
    - model_name (str): Name of the pretrained MusicGen model to use.

    Returns:
    - torch.Tensor: Generated waveform tensor.
    - int: Sample rate of the generated audio.
    """
    assert overlap_duration < segment_duration, "Overlap must be less than segment duration."
    step_size = segment_duration - overlap_duration
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the pretrained MusicGen model
    model = MusicGen.get_pretrained(model_name)
    model.set_generation_params(duration=segment_duration)

    # Initialize
    generated_audio = []
    current_prompt = prompt
    num_segments = (total_duration - overlap_duration) // step_size

    for i in range(num_segments):
        print(f"Generating segment {i + 1}/{num_segments}...")

        if isinstance(current_prompt, str):
            # Generate the initial segment from the text prompt
            segment = model.generate([current_prompt])[0].to(device)
        else:
            # Generate continuation based on the audio prompt
            segment = model.generate_continuation(current_prompt, prompt_sample_rate=model.sample_rate, progress=True)[0].to(device)

        generated_audio.append(segment)

        if i < num_segments - 1:
            # Update prompt for next segment: use the last 'overlap_duration' seconds of the current segment
            current_prompt = segment[:, -overlap_duration * model.sample_rate:]

    # Concatenate all segments
    full_audio = torch.cat(generated_audio, dim=-1)

    return full_audio, model.sample_rate