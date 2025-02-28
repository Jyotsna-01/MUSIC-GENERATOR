# MUSIC-GENERATOR

## Overview
This project utilizes deep learning models such as LSTMs to generate music. The model is trained on MIDI files and produces new musical sequences based on learned patterns.

## Features
- **MIDI Processing**: Converts MIDI files into note sequences for model training.
- **LSTM-Based Model**: Utilizes a deep learning model to predict musical patterns.
- **Music Generation**: Produces new note sequences based on trained data.
- **MIDI Output**: Saves generated music as a playable MIDI file.

## Prerequisites
Ensure you have the following dependencies installed before running the project:
```sh
pip install numpy tensorflow pretty_midi
```

## How to Run the Project
1. **Clone the Repository:**
   ```sh
   git clone https://github.com/your-repo-name/music-generation.git
   cd music-generation
   ```
2. **Prepare the MIDI File:**
   - Place your input MIDI file (`d-callme.mid`) in the correct path (`C:/Users/jyots/Downloads/`).

3. **Run the Python Script:**
   ```sh
   python music_generator.py
   ```
   - The model will train on the MIDI data and generate new music.
   - The output file will be saved as `generated_music.mid`.

## Project Structure
```
├── music_generator.py  # Main script for training and music generation
├── d-callme.mid        # Input MIDI file (Example)
├── generated_music.mid # Output generated music file
├── README.md           # Documentation
└── code                # Folder containing the project code files
```

## License
This project is open-source and can be modified as needed.
