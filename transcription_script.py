import os
import torch
import numpy as np
import warnings
from pydub import AudioSegment
from transformers import pipeline

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)

# Suppress all warnings
warnings.filterwarnings("ignore")

pipe = pipeline(
  "automatic-speech-recognition",
  model="openai/whisper-large-v2",
  chunk_length_s=30,
  device=device,
)

directory = "/home/gagan/samples"

for filename in os.listdir(directory):
    file_path = os.path.join(directory, filename)
    try:
        # Load audio file with pydub
        audio = AudioSegment.from_file(file_path)

        # Convert to numpy array
        audio_arr = np.array(audio.get_array_of_samples())

        # If stereo, take only one channel
        if audio.channels == 2:
            audio_arr = audio_arr[::2]

        # Normalize to floats between -1 and 1, as expected by the pipeline
        audio_input = audio_arr / np.iinfo(audio_arr.dtype).max

        # Transcribe and save the output
        transcription = pipe(audio_input)

        # Save the transcription to a text file
        with open(os.path.join(directory, filename.rsplit(".", 1)[0] + ".txt"), "w") as file:
            file.write(transcription[0]['transcription'])

    except Exception as e:
        print(f"Failed to process {file_path}: {e}")
