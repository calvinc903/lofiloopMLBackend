#!/usr/bin/env python
"""
This file defines a generate_long_music() method that uses the MusicGen model
to generate exactly 4 minutes (240 seconds) of music using a 20-second generation window,
with progress logging and saving the output as a WAV file.
"""

import torch
import torchaudio
from audiocraft.models import MusicGen  # Adjust this import if your package structure differs

def generate_long_music(prompt: str) -> torch.Tensor:
    """
    Generate exactly 4 minutes of music using a 20-second window per generation step,
    while logging progress to the console.

    Args:
        prompt (str): A text string used to condition the generation.

    Returns:
        torch.Tensor: The generated audio tensor of shape [B, C, T].
    """
    # Define a custom progress callback to log progress based on generated audio duration.
    def progress_callback(generated_tokens: int, total_tokens: int):
        # Compute current seconds and percentage based on frame_rate.
        current_seconds = generated_tokens / model.frame_rate
        progress_percentage = (current_seconds / 240) * 100
        print(f"Generated duration: {current_seconds:.2f}s ({progress_percentage:.2f}%)", end='\r')
    
    # Load the pretrained MusicGen model (using the 'melody' variant).
    model = MusicGen.get_pretrained(name="melody")
    
    # Override the maximum duration (window) for a single generation chunk to 20 seconds.
    model.max_duration = 20
    
    # Set generation parameters:
    #   - Total duration of the output audio: 240 seconds (4 minutes).
    #   - extend_stride: 15 seconds (must be less than max_duration).
    model.set_generation_params(duration=240, extend_stride=15)
    
    # Set the custom progress callback.
    model.set_custom_progress_callback(progress_callback)
    
    # Generate audio conditioned on the prompt.
    # The generate() method expects a list of descriptions, so we wrap our prompt.
    generated_audio = model.generate([prompt], progress=True)
    
    # Print a newline after progress logging.
    print()
    
    return generated_audio

def save_audio(audio: torch.Tensor, sample_rate: int, filename: str):
    """
    Save a torch.Tensor representing audio to a WAV file.

    Args:
        audio (torch.Tensor): Audio tensor of shape [B, C, T]. This function saves the first sample.
        sample_rate (int): The sample rate of the audio.
        filename (str): The output filename.
    """
    # Ensure the tensor is on CPU and select the first sample in the batch.
    waveform = audio[0].cpu()  # waveform shape should be [C, T]
    torchaudio.save(filename, waveform, sample_rate)
    print(f"Audio saved to {filename}")

if __name__ == "__main__":
    # Example usage: provide a text prompt to generate long music.
    prompt_text = "Create a chill, melancholic yet relaxing lofi hip-hop song that captures the feeling of a quiet evening in a cozy apartment, with rain gently pattering against the window, evoking warm nostalgia with a slight jazzy influence. The tempo should be 70-85 BPM with a swing quantization for a natural, humanized feel. The drums should be soft and vintage, featuring a warm, dusty kick, a loose rimshot or brushed snare, shuffled hi-hats, and occasional foley sounds like pen taps or pages flipping, all played with a slight offbeat groove. The bass should be a smooth, legato upright or moog-style sine wave, locking into the kick for a subtle, groovy presence. The chords should be rich and jazzy, played on a warm Rhodes or Wurlitzer electric piano, acoustic or nylon guitar, and a softly detuned analog synth pad, using seventh, ninth, and extended jazz voicings with an imperfect, humanized touch. The melody should be dreamy and emotional, played on an electric piano, muted trumpet, or soft synth, with slight imperfections and ghostly reverb tails. To enhance atmosphere, layer in vinyl crackle, field recordings (rain, coffee shop murmurs, city ambiance), tape flutter, and stretched reverb tails, with occasional whispered vocal chops or pitched-down phrases. The song should follow a loose structure, beginning with ambient textures and a filtered melody intro, building into a smooth verse with full drums and an expressive lead, then reaching an emotional chorus with added harmonies and reverb washes, followed by a breakdown with reduced instrumentation and a gentle outro fading into crackle and distant reverb. Mixing should emphasize tape saturation for warmth, low-pass filtering for a nostalgic aesthetic, and sidechain compression for subtle breathing dynamics, with delicate reverb and tape delay on key elements to create an organic, immersive listening experience. The final track should feel imperfect yet intentional, warm, and introspective, perfect for studying, late-night thinking, or rainy-day relaxation, carrying an intimate and nostalgic, handcrafted feel."
    audio = generate_long_music(prompt_text)
    print("Generated audio shape:", audio.shape)
    
    # Save the generated audio.
    # Note: Use the model's sample_rate if available. Here we assume 44100 Hz.
    # You may adjust the sample rate based on your MusicGen model's output.
    sample_rate = 44100
    save_audio(audio, sample_rate, "generated_music.wav")