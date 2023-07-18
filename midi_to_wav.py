import os
import subprocess

input_dir = "F:/GA_midis/test/allmidis/100000/"
output_dir = "F:/GA_midis/test/allmidis/100000/batchwavs/"

# Set some options for fluidsynth
soundfont_path = "gm.sf2" # set the path to your soundfont file if using one
sample_rate = 44100
bits_per_sample = 16
channels = 2

# Loop through all the MIDI files in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".mid") or filename.endswith(".midi"):
        # Set the input and output paths
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + ".wav")

    # Check if the output file already exists
    if os.path.exists(output_path):
        print(f"{filename} already converted")
    else:
        # Run fluidsynth to convert the MIDI file to WAV
        try:
            subprocess.run(["fluidsynth/bin/fluidsynth.exe", "-F", output_path, "-r", str(sample_rate), "-c", str(channels), "-ni", soundfont_path, input_path], check=True)
            print(f"{filename} converted successfully")
        except subprocess.CalledProcessError as e:
            print(f"Error converting {filename}: {e}")
