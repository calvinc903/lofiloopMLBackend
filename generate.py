#!/usr/bin/env python
"""
This file defines a generate_long_music() method that uses the MusicGen model
to generate exactly 4 minutes (240 seconds) of music using a 20-second generation window,
with progress logging and an optional song sample input. The generated audio is saved as a WAV file.
"""

import torch
import torchaudio
from audiocraft.models import MusicGen  # Adjust this import if your package structure differs

def generate_long_music(prompt: str, sample_path: str = None) -> torch.Tensor:
    """
    Generate exactly 4 minutes of music using a 20-second window per generation step,
    optionally conditioned on an input song sample, with progress logging.

    Args:
        prompt (str): A text string used to condition the generation.
        sample_path (str, optional): Path to a song sample file to be used for melody conditioning.
                                     If None, only text conditioning is used.
    Returns:
        torch.Tensor: The generated audio tensor of shape [B, C, T].
    """
    # Define a custom progress callback that converts generated token count into a duration-based percentage.
    def progress_callback(generated_tokens: int, total_tokens: int):
        current_seconds = generated_tokens / model.frame_rate
        progress_percentage = (current_seconds / 240) * 100
        print(f"Generated duration: {current_seconds:.2f}s ({progress_percentage:.2f}%)", end='\r')
    
    # Load the pretrained MusicGen model (using the 'melody' variant).
    model = MusicGen.get_pretrained(name="melody")
    
    # Override the maximum generation window to 20 seconds.
    model.max_duration = 20
    
    # Set generation parameters:
    #   - Total duration: 240 seconds (4 minutes)
    #   - extend_stride: 15 seconds (must be less than max_duration)
    model.set_generation_params(duration=240, extend_stride=15)
    
    # Set the progress callback.
    model.set_custom_progress_callback(progress_callback)
    
    # Generate audio using text conditioning only or with an additional song sample.
    if sample_path is not None:
        # Load the song sample using torchaudio.
        melody_wav, melody_sample_rate = torchaudio.load(sample_path)
        # generate_with_chroma expects a list of text descriptions, a melody sample, and its sample rate.
        generated_audio = model.generate_with_chroma([prompt], melody_wav, melody_sample_rate, progress=True)
    else:
        generated_audio = model.generate([prompt], progress=True)
    
    print()  # Print newline after progress logging.
    return generated_audio

def save_audio(audio: torch.Tensor, sample_rate: int, filename: str):
    """
    Save a torch.Tensor representing audio to a WAV file.

    Args:
        audio (torch.Tensor): Audio tensor of shape [B, C, T]. This function saves the first sample.
        sample_rate (int): The sample rate of the audio.
        filename (str): The output filename.
    """
    waveform = audio[0].cpu()  # Use the first audio sample in the batch.
    torchaudio.save(filename, waveform, sample_rate)
    print(f"Audio saved to {filename}")

if __name__ == "__main__":
    # Example usage:
    prompt_text = "Lofi slow bpm electro chill with organic samples."
    # Provide a valid sample song file path or set to None to use only text conditioning.
    sample_path = "assets/lofi1.mp3"  # Replace with your file path or use None
    
    audio = generate_long_music(prompt_text, sample_path)
    print("Generated audio shape:", audio.shape)
    
    # Save the generated audio. Adjust sample_rate if needed (model's sample rate is preferable).
    sample_rate = 44100
    save_audio(audio, sample_rate, "generated_music.wav")