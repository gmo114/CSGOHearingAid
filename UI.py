##################################################################################################
# IMPORTS

import sys
from threading import *
import time

import tkinter as tk

import numpy as np
import pyaudio

from CNN import NeuralNetwork

##################################################################################################
# VARIABLES & CONSTANTS

# TKInter
DEFAULT_SIZE = 32
DEFAULT_TYPE = 'Arial'
PADDING = 25

root = tk.Tk()
root.title('Direction')
root.geometry('400x300')
root.attributes('-alpha', 0.5)
root.attributes('-topmost', True)

u = '^'
d = 'v'
r = '>'
l = '<'

labels = {
    'front': tk.Label(root, text=u, font=(DEFAULT_TYPE, DEFAULT_SIZE)),
    'back': tk.Label(root, text=d, font=(DEFAULT_TYPE, DEFAULT_SIZE)),
    'right': tk.Label(root, text=r, font=(DEFAULT_TYPE, DEFAULT_SIZE)),
    'left': tk.Label(root, text=l, font=(DEFAULT_TYPE, DEFAULT_SIZE))
}

# Frame for buttons:
buttons_frame = tk.Frame(root)
buttons_frame.pack(side='bottom', pady=(PADDING,0))

# start button
color = 'black'
button = tk.Button(
    buttons_frame, text=f"START",
    command=lambda: threadWrapper()
)
button.pack(side='left', padx=(PADDING,0))

# stop button
color = 'black'
button = tk.Button(
    buttons_frame, text=f"TOGGLE",
    command=lambda: toggle()
)
button.pack(side='left', padx=(PADDING,0))

labels['front'].pack(side='top', pady=(PADDING,0))
labels['back'].pack(side='bottom', pady=(0,PADDING))
labels['right'].pack(side='right', padx=(0,PADDING))
labels['left'].pack(side='left', padx=(PADDING,0))

# PyAudio
FORMAT = pyaudio.paInt16    # Audio format
CHANNELS = 1                # Number of audio channels
RATE = 44100                # Sample Rate
CHUNK = 1024                # Samples per buffer

audio = pyaudio.PyAudio()

# NeuralNetwork

n = NeuralNetwork(fromJson='models/model(1).json')

tags = {
        'left': 0,
        'up': 1,
        'right': 2,
        'down': 3,
        '': -1,
    }

keys = {
        0: 'left',
        1: 'up',
        2: 'right',
        3: 'down',
        -1: '',
    }

##################################################################################################
# FUNCTIONS

# Function to open the audio stream
def open_stream(device_index):
    stream = audio.open(
                            format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            output=False, # capturing audio, so output is set to False.
                            frames_per_buffer=CHUNK,
                            input_device_index=device_index # Specify the device index here.
                        )
    return stream


# Dummy function to analyze the loudness
def analyze_loudness(data):
    try:
        # Convert data to integers
        audio_data = np.frombuffer(data, dtype=np.int16)
        # Calculate the volume as the RMS of the signal
        mean_val = np.mean(np.abs(audio_data)**2)
        if mean_val < 0:
            mean_val *= -1
        volume = np.sqrt(mean_val)
    except:
        volume = -1
    return volume


def analyze(data):
    audio_data = np.frombuffer(data, dtype=np.int16)
    # This will ignore any remainder that doesn't fit into a section of length 5
    num_complete_sections = len(audio_data) // 5
    sections = audio_data[:num_complete_sections * 5].reshape(-1, 5)
    return sections[0]


def toggle():
    global run
    run = not run
    if run:
        print('on')
    else:
        print('off')


run = True


# Takes audio from the selected device and processes it by calling analyze_loudness()
def listen():
    global root
    global run

    if not run:
        run = True
    device_index = 3 # AVAILABLE DEVICE INDEXES CAN BE FOUND ON "audio_devices.txt". This number will be different depending on the machine that it is run
    stream = open_stream(device_index)

    print("Listening...")

    try:
        while run:
            data = stream.read(CHUNK, exception_on_overflow=False)

            audioData = analyze(data)
            outputs = n.query(audioData)

            # TODO change prediction based on your model here
            prediction = np.argmax(outputs)
            prediction = keys[prediction]

            prediction = prediction.lower()

            if prediction.lower() == 'front-left':
                labels['left'].config(font=(DEFAULT_TYPE, DEFAULT_SIZE + (DEFAULT_SIZE//2)), fg='blue')
                labels['front'].config(font=(DEFAULT_TYPE, DEFAULT_SIZE + (DEFAULT_SIZE//2)), fg='blue')
                time.sleep(1)
                labels['left'].config(font=(DEFAULT_TYPE, DEFAULT_SIZE), fg='black')
                labels['front'].config(font=(DEFAULT_TYPE, DEFAULT_SIZE), fg='black')
                time.sleep(1.5)

            elif prediction.lower() == 'left-right':
                labels['left'].config(font=(DEFAULT_TYPE, DEFAULT_SIZE + (DEFAULT_SIZE//2)), fg='blue')
                labels['front'].config(font=(DEFAULT_TYPE, DEFAULT_SIZE + (DEFAULT_SIZE//2)), fg='blue')
                time.sleep(1)
                labels['left'].config(font=(DEFAULT_TYPE, DEFAULT_SIZE), fg='black')
                labels['front'].config(font=(DEFAULT_TYPE, DEFAULT_SIZE), fg='black')
                time.sleep(1.5)
            
            elif prediction.lower() == 'front-right':
                labels['front'].config(font=(DEFAULT_TYPE, DEFAULT_SIZE + (DEFAULT_SIZE//2)), fg='blue')
                labels['right'].config(font=(DEFAULT_TYPE, DEFAULT_SIZE + (DEFAULT_SIZE//2)), fg='blue')
                time.sleep(1)
                labels['front'].config(font=(DEFAULT_TYPE, DEFAULT_SIZE), fg='black')
                labels['right'].config(font=(DEFAULT_TYPE, DEFAULT_SIZE), fg='black')
                time.sleep(1.5)
            
            elif prediction.lower() == 'right-left':
                labels['front'].config(font=(DEFAULT_TYPE, DEFAULT_SIZE + (DEFAULT_SIZE//2)), fg='blue')
                labels['right'].config(font=(DEFAULT_TYPE, DEFAULT_SIZE + (DEFAULT_SIZE//2)), fg='blue')
                time.sleep(1)
                labels['front'].config(font=(DEFAULT_TYPE, DEFAULT_SIZE), fg='black')
                labels['right'].config(font=(DEFAULT_TYPE, DEFAULT_SIZE), fg='black')
                time.sleep(1.5)
            
            elif prediction.lower() == 'right-right':
                labels['right'].config(font=(DEFAULT_TYPE, DEFAULT_SIZE + (DEFAULT_SIZE//2)), fg='blue')
                labels['back'].config(font=(DEFAULT_TYPE, DEFAULT_SIZE + (DEFAULT_SIZE//2)), fg='blue')
                time.sleep(1)
                labels['right'].config(font=(DEFAULT_TYPE, DEFAULT_SIZE), fg='black')
                labels['back'].config(font=(DEFAULT_TYPE, DEFAULT_SIZE), fg='black')
                time.sleep(1.5)
            
            elif prediction.lower() == 'back-right':
                labels['right'].config(font=(DEFAULT_TYPE, DEFAULT_SIZE + (DEFAULT_SIZE//2)), fg='blue')
                labels['back'].config(font=(DEFAULT_TYPE, DEFAULT_SIZE + (DEFAULT_SIZE//2)), fg='blue')
                time.sleep(1)
                labels['right'].config(font=(DEFAULT_TYPE, DEFAULT_SIZE), fg='black')
                labels['back'].config(font=(DEFAULT_TYPE, DEFAULT_SIZE), fg='black')
                time.sleep(1.5)
            
            elif prediction.lower() == 'left-left':
                labels['left'].config(font=(DEFAULT_TYPE, DEFAULT_SIZE + (DEFAULT_SIZE//2)), fg='blue')
                labels['back'].config(font=(DEFAULT_TYPE, DEFAULT_SIZE + (DEFAULT_SIZE//2)), fg='blue')
                time.sleep(1)
                labels['left'].config(font=(DEFAULT_TYPE, DEFAULT_SIZE), fg='black')
                labels['back'].config(font=(DEFAULT_TYPE, DEFAULT_SIZE), fg='black')
                time.sleep(1.5)
            
            elif prediction.lower() == 'back-left':
                labels['left'].config(font=(DEFAULT_TYPE, DEFAULT_SIZE + (DEFAULT_SIZE//2)), fg='blue')
                labels['back'].config(font=(DEFAULT_TYPE, DEFAULT_SIZE + (DEFAULT_SIZE//2)), fg='blue')
                time.sleep(1)
                labels['left'].config(font=(DEFAULT_TYPE, DEFAULT_SIZE), fg='black')
                labels['back'].config(font=(DEFAULT_TYPE, DEFAULT_SIZE), fg='black')
                time.sleep(1.5)
            
            else:
                labels[prediction].config(font=(DEFAULT_TYPE, DEFAULT_SIZE + (DEFAULT_SIZE//2)), fg='blue')
                time.sleep(1)
                labels[prediction].config(font=(DEFAULT_TYPE, DEFAULT_SIZE), fg='black')
                time.sleep(1.5)

    except KeyboardInterrupt:
        print("Program stopped.")

    print("No longer listening.")

    # Close and terminate the stream and audio
    stream.stop_stream()
    stream.close()
    audio.terminate()

##################################################################################################
# MAIN

def threadWrapper():
    t = Thread(target=listen)
    t.start()


def main():
    root.mainloop()


if __name__ == '__main__':
    main()