# Audio to Text Transcription using OpenAI's Whisper ASR Model

This Python script uses OpenAI's Whisper ASR model (via Hugging Face's `transformers` library) to transcribe audio files from a specified directory. 

The script supports any audio format that `pydub` and `ffmpeg` can handle, which includes .wav, .mp3, .flac, and .ogg formats, among others. The transcriptions are written to text files and saved in a separate directory.

## Dependencies
The following Python libraries are required:
- torch
- numpy
- warnings
- pydub
- transformers

Please ensure you have these installed. You can install these using pip:
```
pip install torch numpy pydub transformers
```

The script also requires `ffmpeg`, which can be installed on Ubuntu as follows:
```
sudo apt update
sudo apt install ffmpeg
```

## Usage
The script automatically uses the GPU if available, otherwise it falls back to CPU.

1. Clone the repository and navigate to the directory.

2. Run the script with Python:
   ```
   python transcription_script.py
   ```
The script reads audio files from a directory specified by the `aud_directory` variable, and writes the transcriptions to a directory specified by the `xpt_directory` variable. Modify these paths in the script according to your needs.

Each transcription is saved as a text file with the same name as the corresponding audio file (but with a .txt extension).

## Note
This script suppresses all Python warnings for convenience. However, it is generally recommended to address the cause of warnings rather than suppressing them.
