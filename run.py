from generate_long_music import generate_long_music

# User-defined description
description = "chill happy lofi music"

# Generate music with only a text prompt
output_file = generate_long_music(description, output_path="long_lofi")

print(f"Generated 4-minute track saved at: {output_file}")