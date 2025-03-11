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


def generate_music_continuation(input_wav, model_name='facebook/musicgen-melody', progress=True, output_path=None, max_prompt_duration=10):
    """
    Generates a continuation of the given input lofi audio file.

    Parameters:
    - input_wav (str): Path to the input .wav file (must be at least max_prompt_duration seconds long).
    - model_name (str): Pretrained MusicGen model to use.
    - progress (bool): Whether to show generation progress.
    - output_path (str, optional): Base name for the output file (without extension or with .wav).
      If provided, the generated audio will be saved to this file.
    - max_prompt_duration (int or float): Maximum duration in seconds of the prompt used for continuation.

    Returns:
    - If output_path is provided: str, the path to the saved audio file.
    - Otherwise: Tensor, the generated audio waveform.
    """
    # MusicGen doesn't support .to(), so we run everything on CPU.
    device = "cpu"
    print(f"Using device: {device}")

    # Load the pretrained MusicGen model (remains on CPU)
    model = MusicGen.get_pretrained(model_name)
    
    # Load the input audio
    prompt_waveform, prompt_sr = torchaudio.load(input_wav)
    num_samples = prompt_waveform.shape[1]

    # Calculate the number of samples corresponding to max_prompt_duration
    required_samples = int(max_prompt_duration * prompt_sr)
    if num_samples < required_samples:
        raise ValueError(f"Input audio must be at least {max_prompt_duration} seconds long, but got {num_samples / prompt_sr:.2f} seconds.")

    # Trim the prompt to the last max_prompt_duration seconds
    prompt_waveform = prompt_waveform[:, -required_samples:]

    # Generate continuation (the prompt tokens should now be within the acceptable range)
    output = model.generate_continuation(prompt_waveform, prompt_sample_rate=prompt_sr, progress=progress)
    
    # Optionally display the generated audio if a display function is defined.
    try:
        display_audio(output, sample_rate=32000)
    except NameError:
        pass  # display_audio is not defined

    # If an output_path is provided, save the generated audio to a WAV file.
    if output_path:
        if not output_path.endswith(".wav"):
            output_path = f"{output_path}.wav"
        sample_rate = getattr(model, "sample_rate", 32000)
        torchaudio.save(output_path, output.cpu().detach(), sample_rate)
        return output_path

    return output