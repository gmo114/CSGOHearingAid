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
    'up': tk.Label(root, text=u, font=(DEFAULT_TYPE, DEFAULT_SIZE)),
    'down': tk.Label(root, text=d, font=(DEFAULT_TYPE, DEFAULT_SIZE)),
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

labels['up'].pack(side='top', pady=(PADDING,0))
labels['down'].pack(side='bottom', pady=(0,PADDING))
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


# TODO create our own function that processes the audio to determine the direction of any nearby enemies
def analyze(data):
    audio_data = np.frombuffer(data, dtype=np.int16)
    # This will ignore any remainder that doesn't fit into a section of length 5
    num_complete_sections = len(audio_data) // 5
    sections = audio_data[:num_complete_sections * 5].reshape(-1, 5)
    return sections[0]

    # if len(audio_data) % 5 == 0:
    #     reshaped_data = audio_data.reshape(-1, 5)
    # else:
    #     print(f' > INVALID DATA')
    #     sys.exit(-1)


def toggle():
    global run
    run = not run
    if run:
        print('on')
    else:
        print('off')


run = True

# dataLeft = 0

# Takes audio from the selected device and processes it by calling analyze_loudness()
def listen():
    global root
    global run
    global dataLeft

    if not run:
        run = True
    device_index = 4 # AVAILABLE DEVICE INDEXES CAN BE FOUND ON "audio_devices.txt". This number will be different depending on the machine that it is run
    stream = open_stream(device_index)

    print("Listening...")

    try:
        while run:
            data = stream.read(CHUNK, exception_on_overflow=False)

            # if dataLeft <= 0:
            audioData = analyze(data)
                # dataLeft = len(audioData)
            # else:
                # for inputData in audioData:
                #     if not run:
                #         break
                    # dataLeft -= 1
            outputs = n.query(audioData)
            # the index of the highest value corresponds to the prediction
            prediction = np.argmax(outputs)
            prediction = keys[prediction]

            # volume = analyze_loudness(data) # TODO here is where our analyze() function must go, which determines the direction of the enemies

            # TODO change the 'if' below to correctly check for the direction depending on our analyze() function
            # if volume > 50:  # This is an arbitrary threshold; adjust based on your needs
                # print('I heard that!')
            labels[prediction].config(font=(DEFAULT_TYPE, DEFAULT_SIZE + (DEFAULT_SIZE//2)))
            print(f"Someone's on my {prediction}!")
            time.sleep(1)
            labels[prediction].config(font=(DEFAULT_TYPE, DEFAULT_SIZE))
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