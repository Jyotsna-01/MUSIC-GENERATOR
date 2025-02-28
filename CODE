import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation, Embedding
from tensorflow.keras.optimizers import Adam
import pretty_midi

# Helper function to read MIDI files and convert them into note sequences
def midi_to_note_sequence(midi_file):
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    notes = []
    for instrument in midi_data.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                notes.append((note.start, note.pitch, note.end))
    return sorted(notes)

# Tokenize the notes into sequences
def tokenize_notes(notes, seq_length):
    pitches = [note[1] for note in notes]
    unique_pitches = sorted(list(set(pitches)))
    pitch_to_int = {pitch: i for i, pitch in enumerate(unique_pitches)}

    sequences = []
    targets = []
    for i in range(0, len(pitches) - seq_length):
        seq_in = pitches[i:i + seq_length]
        seq_out = pitches[i + seq_length]
        sequences.append([pitch_to_int[note] for note in seq_in])
        targets.append(pitch_to_int[seq_out])
        
    return np.array(sequences), np.array(targets), len(unique_pitches), pitch_to_int

# Build the model architecture
def build_model(seq_length, num_pitches):
    model = Sequential()
    model.add(Embedding(num_pitches, 100, input_length=seq_length))
    model.add(LSTM(256, return_sequences=True))
    model.add(LSTM(256))
    model.add(Dense(num_pitches))
    model.add(Activation('softmax'))
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer)
    return model

# Generate music using the trained model
def generate_music(model, seed_sequence, num_notes, pitch_to_int, int_to_pitch):
    generated_sequence = list(seed_sequence)
    for _ in range(num_notes):
        input_seq = np.reshape(generated_sequence[-len(seed_sequence):], (1, len(seed_sequence)))
        prediction = model.predict(input_seq, verbose=0)
        next_note = np.argmax(prediction)
        generated_sequence.append(next_note)
    return [int_to_pitch[i] for i in generated_sequence]

# Save the generated sequence to a MIDI file and print the saved location
def sequence_to_midi(sequence, output_file):
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    start_time = 0
    for pitch in sequence:
        note = pretty_midi.Note(velocity=100, pitch=pitch, start=start_time, end=start_time + 0.5)
        instrument.notes.append(note)
        start_time += 0.5
    midi.instruments.append(instrument)
    midi.write(output_file)
    
    # Print confirmation message to ensure the file was saved
    print(f'MIDI file saved at: {output_file}')

# Main function to run the project
def main():
    midi_file = 'C:/Users/jyots/Downloads/d-callme.mid'  # Replace with your MIDI file path
    notes = midi_to_note_sequence(midi_file)
    seq_length = 20
    sequences, targets, num_pitches, pitch_to_int = tokenize_notes(notes, seq_length)
    
    int_to_pitch = {i: pitch for pitch, i in pitch_to_int.items()}
    
    model = build_model(seq_length, num_pitches)
    model.fit(sequences, targets, epochs=10, batch_size=64)
    
    seed_sequence = sequences[0]
    generated_notes = generate_music(model, seed_sequence, 100, pitch_to_int, int_to_pitch)
    
    # Specify where you want the file to be saved
    output_file = 'C:/Users/jyots/Downloads/generated_music.mid'  # Specify path to save generated music
    sequence_to_midi(generated_notes, output_file)

if __name__ == '__main__':
    main()
