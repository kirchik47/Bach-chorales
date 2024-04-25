import numpy as np
from tensorflow import keras
import tensorflow as tf
import pandas as pd
import os
from math import inf
from midiutil import MIDIFile
import pygame


filepath = 'jsb_chorales'

def get_set(type):
    full_path = os.path.join(filepath, type)
    arr = []
    for path, dirs, files in os.walk(full_path):
        for file_name in files:
            df = pd.read_csv(os.path.join(path, file_name))
            arr.append(df.values.tolist())
    return arr

def get_labels(X, dataset):
    y = []
    for i in range(len(X)):
        instance = np.empty(shape=X[i].shape + (4,))
        for steps_ahead in range(1, 5):
            instance[..., steps_ahead - 1] = dataset[i][steps_ahead:steps_ahead+dataset[i].shape[0]-4, :]
        y.append(instance)
    return y

def play_notes(chorale):    
    degrees = chorale
    print(degrees)
    track = 0
    channel = 0
    time = 0 
    duration = 1
    tempo = 60 
    volume = 100 
    MyMIDI = MIDIFile(1) 
    MyMIDI.addTempo(track,time, tempo)
    for pitches in degrees:
        for pitch in pitches:
            MyMIDI.addNote(track, channel, pitch, time, duration, volume)
            time = time + 0.2

    with open("final_melody.mid", "wb") as output_file:
        MyMIDI.writeFile(output_file)

    freq = 44100
    bitsize = -16  
    channels = 2  
    buffer = 1024  
    pygame.mixer.init(freq, bitsize, channels, buffer)
    pygame.mixer.music.set_volume(0.8)
    clock = pygame.time.Clock()
    pygame.mixer.music.load('final_melody.mid')
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        clock.tick(30)

def get_targets(batch):
    X = batch[:, :-1]
    y = batch[:, 1:]
    return X, y

def preprocess(window):
    window = tf.where(window == 0, window, window - min_note + 1)
    return tf.reshape(window, [-1])

def bach_dataset(dataset, batch_size=32, window_size=32, shift=16, shuffle_buffer_size=None, cache=False, seed=42):
    def batch_window(window):
        return window.batch(window_size + 1)
    
    def make_window(chorale):
        dataset = tf.data.Dataset.from_tensor_slices(chorale)
        dataset = dataset.window(window_size + 1, shift, drop_remainder=True)
        return dataset.flat_map(batch_window)

    dataset = tf.ragged.constant(dataset, ragged_rank=1)
    dataset = tf.data.Dataset.from_tensor_slices(dataset)
    dataset = dataset.flat_map(make_window).map(preprocess)
    # for instance in dataset:
    #     print(instance)
    if cache:
        dataset = dataset.cache()
    if shuffle_buffer_size:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size, seed=seed)
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.map(get_targets)
    return dataset.prefetch(1)
    
min_note = inf
max_note = 0
train_set = get_set('train')
for chorale in train_set:
    for notes in chorale:
        prev_min = min_note
        min_note = min(min_note, min(notes))
        if min_note == 0:
            min_note = prev_min
        max_note = max(max_note, max(notes))
n_notes = max_note - min_note + 2
train_set = bach_dataset(get_set('train'))
valid_set = bach_dataset(get_set('valid'))
test_set = bach_dataset(get_set('test'))
model = keras.models.Sequential([keras.layers.Embedding(input_dim=n_notes, output_dim=5),
                                 keras.layers.Conv1D(filters=32, kernel_size=2, padding='causal', activation='relu', dilation_rate=1),
                                 keras.layers.BatchNormalization(),
                                 keras.layers.Conv1D(filters=48, kernel_size=2, padding='causal', activation='relu', dilation_rate=2),
                                 keras.layers.BatchNormalization(),
                                 keras.layers.Conv1D(filters=64, kernel_size=2, padding='causal', activation='relu', dilation_rate=4),
                                 keras.layers.BatchNormalization(),
                                 keras.layers.Conv1D(filters=96, kernel_size=2, padding='causal', activation='relu', dilation_rate=8),
                                 keras.layers.BatchNormalization(),
                                 keras.layers.LSTM(256, return_sequences=True),
                                 keras.layers.Dense(n_notes, activation='softmax')])
print(model.summary())
# model.compile(loss='sparse_categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
model = keras.models.load_model('bach.keras')
model.fit(train_set, epochs=10, validation_data=valid_set)
model.save('bach.keras')
new_set = get_set('test')
new_chorale = tf.constant(new_set[0][:30], dtype=tf.int64)
arpegio = preprocess(new_chorale)
arpegio = tf.reshape(arpegio, (1, -1))
print(new_chorale)
for chord in new_chorale:
    for note in chord:
        new_note_probas = model.predict(arpegio)[:, -1]
        logits = tf.math.log(new_note_probas)
        # print(logits)
        # print(tf.expand_dims(new_note, axis=0))
        new_note = tf.random.categorical(logits, num_samples=1)
        # print(new_note)
        arpegio = tf.concat([arpegio, new_note], axis=-1)
        print(arpegio.shape)
    
arpegio = tf.where(arpegio == 0, arpegio, arpegio + min_note - 1)
play_notes(tf.reshape(arpegio, (-1, 4)))
