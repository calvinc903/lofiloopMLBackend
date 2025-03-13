from old.trackModification import concatenate_tracks


final_track = concatenate_tracks(["text_generated_track.wav", "continued_music.wav"], output_path="final_combined_lofi")

print(f"Final combined track saved at: {final_track}")