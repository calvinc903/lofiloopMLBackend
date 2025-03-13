import torchaudio
import torch
import time
import numpy as np
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

def generate_long_music(description, output_path="long_music", total_duration=240, chunk_size=30, overlap=20):
    """
    Generates a long music track using MusicGen with a sliding window approach.

    Parameters:
    - description (str): The text description of the music.
    - output_path (str): Path to save the generated output.
    - total_duration (int): Total duration of the track in seconds (default 4 minutes).
    - chunk_size (int): Size of each generated chunk (default 30 seconds).
    - overlap (int): Overlap duration to maintain continuity (default 20 seconds).
    
    Returns:
    - str: Path to the saved generated audio file.
    """
    # Check for GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load the pretrained model
    model = MusicGen.get_pretrained('facebook/musicgen-melody')
    model.set_generation_params(duration=chunk_size)  # Each chunk is 30 sec

    generated_audio = []  # Store all chunks

    # Start timing
    start_time = time.time()

    # Generate first chunk
    wav = model.generate([description])
    generated_audio.append(wav[0].cpu())

    # Determine how many iterations we need
    num_chunks = (total_duration - chunk_size) // (chunk_size - overlap) + 1

    for i in range(1, num_chunks):
        print(f"Generating chunk {i+1}/{num_chunks}...")

        # Use the last 20 seconds of the previous chunk as context
        context = generated_audio[-1][:, -overlap * model.sample_rate:]

        # Generate the next 30-second chunk
        wav = model.generate([description], prompt=context)
        generated_audio.append(wav[0].cpu())

    # End timing
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total music generation took {elapsed_time:.2f} seconds.")

    # Concatenate all generated chunks
    final_audio = torch.cat(generated_audio, dim=1)

    # Save the final generated audio clip
    output_file = f"{output_path}"
    audio_write(output_file, final_audio, model.sample_rate, strategy="loudness")

    print(f"Generated track saved at: {output_file}.wav")
    return output_file