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
    prompt_text = "Create a soft, melancholic yet soothing lofi hip-hop track, reminiscent of Lofi Girl’s signature study beats, with a warm, nostalgic atmosphere and a slight jazzy influence, evoking the feeling of studying late at night by a dimly lit desk, with rain softly tapping against the window. The tempo should be between 70-85 BPM, featuring swing quantization for a humanized rhythm. The drums should be gentle and vintage, with a soft, warm kick, loose rimshot snare, shuffled hi-hats, and light percussion elements like shakers or subtle foley sounds. A deep, smooth bassline, either an upright bass with subtle slides or a mellow sine wave bass, should lock into the groove for a laid-back but present rhythm. The chord progression should use rich, jazzy voicings (such as seventh, ninth, or suspended chords) played on a Rhodes or Wurlitzer electric piano, with slight timing imperfections for an organic feel. The melody should be minimal and expressive, played on a soft synth, muted trumpet, or lightly plucked guitar, with gentle vibrato and subtle delay for a dreamy touch. To enhance the atmosphere, include vinyl crackle, field recordings (rainfall, city ambiance, distant chatter), and soft analog synth pads, and optionally add pitched-down vocal chops or whispered phrases (e.g., “keep going” or “it’s okay to take a break”) processed with reverb and tape delay for an ethereal effect. The song structure should be fluid and immersive, starting with a soft ambient intro, moving into a smooth verse establishing the groove, expanding with a chorus that introduces subtle melodic variations, breaking down into a drumless section for space, and fading out with rain and vinyl crackles. The mixing should emphasize warmth and depth, applying tape saturation for vintage character, a low-pass filter to soften high frequencies, and sidechain compression on pads and bass for a breathing effect. Overall, the track should feel intimate, nostalgic, and effortlessly calming, making it the perfect background for studying, late-night relaxation, or introspective moments, seamlessly fitting into the Lofi Girl aesthetic."
    audio = generate_long_music(prompt_text)
    print("Generated audio shape:", audio.shape)
    
    # Save the generated audio.
    # Note: Use the model's sample_rate if available. Here we assume 44100 Hz.
    # You may adjust the sample rate based on your MusicGen model's output.
    sample_rate = 44100
    save_audio(audio, sample_rate, "generated_music.wav")