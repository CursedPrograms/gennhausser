import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from music21 import stream, note, midi
from datetime import datetime
import json
import os

# Paths
SETTINGS_PATH = 'settings.json'
OUTPUT_MODEL_DIR = 'output/models'
OUTPUT_MIDI_DIR = 'output/midi'
FINAL_MODELS_DIR = 'output/final_models'

# Function to get user input
def get_user_input():
    try:
        epochs = int(input("Enter the number of epochs: "))
        batch_size = int(input("Enter the batch size: "))
        length = int(input("Enter the length of the generated sequence: "))
        temperature = float(input("Enter the temperature for randomness: "))
        return epochs, batch_size, length, temperature
    except ValueError:
        print("Invalid input. Please enter valid numeric values.")
        return get_user_input()

# Load settings from JSON
with open(SETTINGS_PATH) as f:
    settings = json.load(f)

generate_from_settings = input("Generate from settings.json? (y/n): ").lower() == 'y'

# Get user input or use settings from JSON
if not generate_from_settings:
    epochs, batch_size, length, temperature = get_user_input()
else:
    epochs = settings.get("epochs")
    batch_size = settings.get("batch_size")
    length = settings.get("length")
    temperature = settings.get("temperature")

# Function to generate a synthetic dataset
def generate_dataset(num_samples=1000, sequence_length=50, num_classes=128):
    X = np.random.random((num_samples, sequence_length, 1))
    y = np.random.randint(0, num_classes, (num_samples, sequence_length))
    return {'input': X, 'output': y}

# Function to convert a generated sequence to a music21 stream
def sequence_to_stream(sequence):
    music_stream = stream.Stream()

    for note_index in sequence[:, 0]:
        music_stream.append(note.Note(note_index))

    return music_stream

# Function to generate music using the trained model
def generate_music(model, seed_sequence, length=1000, temperature=1.0):
    generated_sequence = seed_sequence.copy()

    for _ in range(length):
        # Predict the next set of notes
        next_notes_prob = model.predict(np.expand_dims(generated_sequence, axis=0))

        # Apply temperature to the probabilities for more or less randomness
        next_notes_prob = np.log(next_notes_prob) / temperature
        next_notes_prob = np.exp(next_notes_prob) / np.sum(np.exp(next_notes_prob))

        # Sample the next notes based on the adjusted probabilities
        next_notes = np.random.choice(output_shape, p=next_notes_prob[0])

        # Append the next notes to the generated sequence
        generated_sequence = np.vstack([generated_sequence, [next_notes]])
        generated_sequence = generated_sequence[-input_shape[0]:]  # Keep only the last 'input_shape[0]' elements

    return generated_sequence

# Function to save a music21 stream to a MIDI file with a timestamp
def save_to_midi(stream, prefix='output/generated_output'):
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    filename = f'{prefix}_{timestamp}.mid'
    
    mf = midi.translate.music21ObjectToMidiFile(stream)
    mf.open(filename, 'wb')
    mf.write()
    mf.close()
    
    return filename

# Function to build the LSTM model
def build_model(input_shape, output_shape):
    model = Sequential()
    model.add(LSTM(256, input_shape=input_shape, return_sequences=True))
    model.add(LSTM(256))
    model.add(Dense(output_shape, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    return model

# Assume 'dataset' is your generated or real music dataset
dataset = generate_dataset()
X, y = dataset['input'], dataset['output']

# Assume 'model' is your trained LSTM model
# Make sure to replace 'input_shape' and 'output_shape' with the actual shapes from your dataset
input_shape = X.shape[1:]
output_shape = y.shape[1]

# Build the LSTM model
model = build_model(input_shape, output_shape)

# Callback to save model
# Define the function to generate music using the trained model
def generate_and_save_music(model, length, temperature, output_dir):
    seed_sequence = X[np.random.randint(0, X.shape[0])]
    generated_music_sequence = generate_music(model, seed_sequence, length=length, temperature=temperature)
    generated_music_stream = sequence_to_stream(generated_music_sequence)
    midi_filename = save_to_midi(generated_music_stream, f'{output_dir}/generated_output')
    print(f'Generated music saved to: {midi_filename}')

# Define the callback class
class SaveModelCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 == 0:
            os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True) 
            model_filename = f'{OUTPUT_MODEL_DIR}/trained_model_epoch_{epoch}.keras'
            self.model.save(model_filename)
            print(f'Model saved as {model_filename} at epoch {epoch}.')

            # Load the saved model
            loaded_model = tf.keras.models.load_model(model_filename)

            # Generate and save music with the loaded model
            generate_and_save_music(loaded_model, length, temperature, OUTPUT_MIDI_DIR)

# Train the model
model.fit(X, y, epochs=epochs, batch_size=batch_size, callbacks=[SaveModelCallback()])

# Save the trained model with a timestamp
model_timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
model.save(f'{FINAL_MODELS_DIR}/trained_model_{model_timestamp}.keras')
print(f'Model saved as {FINAL_MODELS_DIR}/trained_model_{model_timestamp}.keras.')
loaded_model = tf.keras.models.load_model(f'{FINAL_MODELS_DIR}/trained_model_{model_timestamp}.keras')
generate_and_save_music(loaded_model, length, temperature, OUTPUT_MIDI_DIR)

