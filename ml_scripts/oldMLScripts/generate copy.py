import torch
from audiocraft.models import MusicGen
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