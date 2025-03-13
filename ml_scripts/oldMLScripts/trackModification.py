import torch
import torchaudio

def concatenate_tracks(track_paths, output_path="combined_track"):
    """
    Concatenate multiple audio tracks sequentially.
    
    This function assumes that all input tracks have the same number of channels 
    and sample rate. It loads each track, concatenates them along the time axis, 
    and saves the resulting track to a WAV file.
    
    Args:
        track_paths (List[str]): List of file paths to the audio tracks.
        output_path (str): Base name for the output file (without extension or with .wav).
    
    Returns:
        str: File path of the saved concatenated audio track.
    """
    tracks = []
    sample_rates = []

    # Load all tracks
    for path in track_paths:
        waveform, sr = torchaudio.load(path)
        tracks.append(waveform)
        sample_rates.append(sr)
    
    # Ensure all tracks have the same sample rate
    if len(set(sample_rates)) != 1:
        raise ValueError("Not all tracks have the same sample rate. Resampling is required.")
    sr = sample_rates[0]
    
    # Ensure all tracks have the same number of channels
    channels = [track.shape[0] for track in tracks]
    if len(set(channels)) != 1:
        raise ValueError("Not all tracks have the same number of channels.")
    
    # Concatenate the tracks along the time dimension (dim=1)
    combined = torch.cat(tracks, dim=1)
    
    # Ensure output_path ends with .wav
    output_file = output_path if output_path.endswith(".wav") else f"{output_path}.wav"
    
    # Save the concatenated track to disk
    torchaudio.save(output_file, combined, sr)
    
    return output_file