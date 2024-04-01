##################################################################################################
# IMPORTS

from threading import *
import time

import tkinter as tk

import numpy as np
import pyaudio

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
CHANNELS = 2                # Number of audio channels
RATE = 44100                # Sample Rate
CHUNK = 1024                # Samples per buffer

audio = pyaudio.PyAudio()

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
def get_direction(data):
    pass


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
            volume = analyze_loudness(data) # TODO here is where our analyze() function must go, which determines the direction of the enemies

            # TODO change the 'if' below to correctly check for the direction depending on our analyze() function
            if volume > 50:  # This is an arbitrary threshold; adjust based on your needs
                print('I heard that!')
                labels['down'].config(font=(DEFAULT_TYPE, DEFAULT_SIZE + (DEFAULT_SIZE//2)))
                time.sleep(1)
                labels['down'].config(font=(DEFAULT_TYPE, DEFAULT_SIZE))
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