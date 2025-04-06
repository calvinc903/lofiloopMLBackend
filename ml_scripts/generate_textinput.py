#!/usr/bin/env python
"""
Improved MusicGen long generation with dynamic prompt adjustment, cosine crossfade,
and optional noise injection for smoother transitions.
Generates exactly 4 minutes (240 seconds) of music in 30-second chunks.
"""

import os
import random
import torch
import torchaudio
import numpy as np

from audiocraft.models import MusicGen

# Global dictionary to track generation progress per song.
song_progress = {}

def set_seed(seed: int = 0):
    """Set the random seed for reproducibility."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if seed <= 0:
        seed = np.random.default_rng().integers(1, 2**32 - 1)
    seed = int(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return seed

def load_model():
    """Load the MusicGen model (using the melody variant)."""
    model = MusicGen.get_pretrained(name="facebook/musicgen-melody")
    return model

def progress_callback(generated_tokens: int, total_tokens: int, song_id: str, model):
    """Update and print generation progress."""
    current_seconds = generated_tokens / model.frame_rate
    progress_percentage = (current_seconds / 240) * 100
    song_progress[song_id] = {
        'duration': current_seconds,
        'progress': progress_percentage
    }
    print(f"Generated duration: {current_seconds:.2f}s ({progress_percentage:.2f}%)", end="\r")

def initial_generate(model, prompt: str, generation_window: int):
    """Generate the initial audio chunk using text-only conditioning."""
    audio = model.generate(descriptions=[prompt], progress=False)
    sr = model.sample_rate
    return audio[:, :, : sr * generation_window]

def cosine_crossfade(audio1, audio2, overlap_length, noise_injection=0.0):
    """
    Apply cosine crossfade between two audio tensors with optional noise injection.
    
    Args:
        audio1 (torch.Tensor): Previous audio [B, C, T]
        audio2 (torch.Tensor): New audio [B, C, T]
        overlap_length (int): Overlap length in samples.
        noise_injection (float): Standard deviation of noise to add during crossfade.
    
    Returns:
        torch.Tensor: Blended audio of shape [B, C, T1 + T2 - overlap_length]
    """
    # Create cosine fade curves.
    fade_out = 0.5 * (1 + torch.cos(torch.linspace(0, torch.pi, steps=overlap_length, device=audio1.device)))
    fade_in = 1 - fade_out

    # Optionally add a small amount of noise to the fade_in curve.
    if noise_injection > 0:
        noise = torch.randn_like(audio1[:, :, -overlap_length:]) * noise_injection
        fade_in = fade_in + noise
        fade_in = torch.clamp(fade_in, 0, 1)  # Ensure values stay between 0 and 1.

    # Apply fades to the overlap regions.
    audio1_overlap = audio1[:, :, -overlap_length:] * fade_out
    audio2_overlap = audio2[:, :, :overlap_length] * fade_in

    blended_overlap = audio1_overlap + audio2_overlap

    # Concatenate non-overlapping parts with the blended overlap.
    result = torch.cat([
        audio1[:, :, :-overlap_length],
        blended_overlap,
        audio2[:, :, overlap_length:]
    ], dim=2)
    return result

def generate_long_music(prompt: str, song_id: str) -> torch.Tensor:
    """
    Generate 4 minutes (240 seconds) of music in 30-second chunks with improved transitions.
    
    Uses a dynamically adjusted prompt (3-second overlap) and cosine crossfade.
    """
    set_seed(42)
    model = load_model()
    sr = model.sample_rate

    final_duration = 240         # Total duration in seconds (4 minutes)
    generation_window = 30       # Each generation step is 30 seconds
    overlap_seconds = 3          # Overlap duration for crossfade and as prompt for continuation
    overlap_samples = sr * overlap_seconds

    # Configure the model's generation parameters.
    model.set_generation_params(
        use_sampling=True,
        top_k=250,
        top_p=0,
        temperature=1.0,
        cfg_coef=3.0,
        duration=generation_window,
    )

    # Set custom progress callback.
    model.set_custom_progress_callback(lambda gen, tot: progress_callback(gen, tot, song_id, model))

    print(f"Starting long-form generation for song '{song_id}'")
    print(f"Generation window: {generation_window}s, Overlap/Prompt: {overlap_seconds}s (cosine crossfade)")

    # Generate the initial audio chunk.
    audio = initial_generate(model, prompt, generation_window)
    current_duration = generation_window
    # Calculate how many additional iterations we need.
    total_iterations = (final_duration - generation_window) // generation_window

    print(f"Initial segment generated: {current_duration}s")

    for step in range(total_iterations):
        # Use only the last 'overlap_seconds' of the current audio as the prompt for continuation.
        previous_chunk = audio[:, :, -sr * overlap_seconds:]
        
        new_chunk = model.generate_continuation(
            previous_chunk,
            descriptions=[prompt],
            prompt_sample_rate=sr,
            progress=False,
        )
        new_segment = new_chunk[:, :, -sr * generation_window:]

        # Apply cosine crossfade with optional noise injection (set noise_injection to 0.0 to disable).
        audio = cosine_crossfade(audio, new_segment, overlap_samples, noise_injection=0.0001)

        current_duration = audio.shape[2] / sr
        progress = (current_duration / final_duration) * 100
        print(f"Progress: {current_duration:3.0f}s generated ({progress:5.1f}%)", end="\r")

    print()  # Move to new line after progress logging.
    total_samples = sr * final_duration
    audio = audio[:, :, :total_samples]
    print(f"Finished generation for song '{song_id}': {final_duration}s total.")
    return audio

def save_audio(audio: torch.Tensor, sample_rate: int, filename: str):
    """
    Save the generated audio tensor as a WAV file.
    """
    waveform = audio[0].cpu()
    torchaudio.save(filename, waveform, sample_rate)
    print(f"Audio saved to {filename}")

# Example usage:
if __name__ == "__main__":
    prompt_text = "Lofi Song with a soft, melancholic yet soothing lofi hip-hop track"
    song_id = "song_001"

    audio = generate_long_music(prompt_text, song_id)
    print("Generated audio shape:", audio.shape)

    model = load_model()
    sr = model.sample_rate
    output_filename = "generated_music.wav"
    save_audio(audio, sr, output_filename)