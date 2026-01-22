# speech-command-recognition

This project explores speech command recognition using short audio clips.
The goal is to take raw .wav files, convert them into time–frequency features, and train a simple neural network to classify spoken commands.

Rather than focusing only on model accuracy, this project also puts some emphasis on data handling and robustness, since real-world audio datasets are often messy.

What this project does

1. Loads short speech recordings from .wav files

2. Converts audio into log-mel spectrograms

3. Trains a small CNN-based classifier

4. Supports running from the command line

5. Handles missing or broken audio files without crashing

