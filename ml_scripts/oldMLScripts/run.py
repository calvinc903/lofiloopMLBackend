def generate_long_music(prompt: str):
    """
    Generate exactly 4 minutes (240 seconds) of music conditioned on a text prompt,
    using a 20-second generation window.
    
    Args:
        prompt (str): The text prompt to condition music generation.
    
    Returns:
        torch.Tensor: A tensor containing the generated audio waveform.
        
    Note:
        This function assumes that the MusicGen class (and its dependencies)
        are available in the import path.
    """
    # Get the pretrained model. Here we use the "melody" model and let it select the device.
    model = MusicGen.get_pretrained(name="melody", device="cuda")
    
    # Override the maximum generation duration (window) to 20 seconds.
    model.max_duration = 20  # 20-second generation window
    
    # Set generation parameters:
    # - duration: total desired generation length (240 seconds = 4 minutes)
    # - extend_stride: stride for iterative generation (here we use 18 seconds so that there is slight overlap).
    model.set_generation_params(duration=240, two_step_cfg=False, extend_stride=18)
    
    # Generate audio conditioned on the prompt. The generate() method expects a list of prompts.
    audio = model.generate([prompt], progress=True)
    
    return audio


# If run.py is executed as the main program, you can add a simple test:
if __name__ == "__main__":
    # Example prompt – change as desired.
    prompt_text = "A soothing, ambient melody with gentle piano and soft strings."
    generated_audio = generate_long_music(prompt_text)
    
    # At this point, 'generated_audio' is a torch.Tensor containing the audio waveform.
    # For example, you might want to save it to a file using your preferred audio library.
    # (No additional imports are used in this file.)
    print("Long music generation complete.")