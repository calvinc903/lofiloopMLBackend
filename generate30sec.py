import torchaudio
import torch
import time
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

def generate_music(description, melody_path, output_path="output", duration=30):
    """
    Generates music based on a given description and melody.
    
    Parameters:
    - description (str): The text description of the music.
    - melody_path (str): Path to the melody audio file.
    - output_path (str): Path to save the generated output.
    - duration (int): Duration of generated audio in seconds.
    
    Returns:
    - str: Path to the saved generated audio file.
    """
    # Check for GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load the pretrained model
    model = MusicGen.get_pretrained('facebook/musicgen-melody')
    model.set_generation_params(duration=duration)  # Set generation duration

    # Load the melody and move it to the correct device
    melody, sr = torchaudio.load(melody_path)
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
    output_file = f"{output_path}.wav"
    audio_write(output_file, wav[0].cpu(), model.sample_rate, strategy="loudness")

    print(f"Generated track saved at: {output_file}")
    return output_file