##################################################################################################
# IMPORTS

from threading import *
import time

import tkinter as tk
from digest import digest
import numpy as np
import pyaudio
import librosa
import joblib
import wave

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
def analyze_loudness(stream):
    temp_wav_file = "temp_audio.wav"
    FORMAT = pyaudio.paInt16  # Audio format (16-bit PCM)
    CHANNELS = 2  # Stereo audio
    RATE = 44100  # Sample rate (Hz)
    CHUNK = 1024
    frames = []
    for _ in range(int(RATE / CHUNK * 2)):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)


    with wave.open(temp_wav_file, 'wb') as wav_file:
        wav_file.setnchannels(CHANNELS)
        wav_file.setsampwidth(audio.get_sample_size(FORMAT))
        wav_file.setframerate(RATE)
        wav_file.writeframes(b''.join(frames))

    data_proccessing = digest()
    return data_proccessing.extract_features("./",temp_wav_file)





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
    knn = joblib.load('knn_model.pkl')
    if not run:
        run = True
    device_index = 3 # AVAILABLE DEVICE INDEXES CAN BE FOUND ON "audio_devices.txt". This number will be different depending on the machine that it is run
    stream = open_stream(device_index)

    print("Listening...")

    try:
        while run:
    
            volume = analyze_loudness(stream) # TODO here is where our analyze() function must go, which determines the direction of the enemies

            # TODO change the 'if' below to correctly check for the direction depending on our analyze() function
            prediction = knn.predict([volume])[0]

            prediction = prediction.lower()

            if prediction.lower() == 'front_left':
                labels['left'].config(font=(DEFAULT_TYPE, DEFAULT_SIZE + (DEFAULT_SIZE//2)), fg='blue')
                labels['front'].config(font=(DEFAULT_TYPE, DEFAULT_SIZE + (DEFAULT_SIZE//2)), fg='blue')
                time.sleep(1)
                labels['left'].config(font=(DEFAULT_TYPE, DEFAULT_SIZE), fg='black')
                labels['front'].config(font=(DEFAULT_TYPE, DEFAULT_SIZE), fg='black')
                time.sleep(1.5)

            elif prediction.lower() == 'left_right':
                labels['left'].config(font=(DEFAULT_TYPE, DEFAULT_SIZE + (DEFAULT_SIZE//2)), fg='blue')
                labels['front'].config(font=(DEFAULT_TYPE, DEFAULT_SIZE + (DEFAULT_SIZE//2)), fg='blue')
                time.sleep(1)
                labels['left'].config(font=(DEFAULT_TYPE, DEFAULT_SIZE), fg='black')
                labels['front'].config(font=(DEFAULT_TYPE, DEFAULT_SIZE), fg='black')
                time.sleep(1.5)
            
            elif prediction.lower() == 'front_right':
                labels['front'].config(font=(DEFAULT_TYPE, DEFAULT_SIZE + (DEFAULT_SIZE//2)), fg='blue')
                labels['right'].config(font=(DEFAULT_TYPE, DEFAULT_SIZE + (DEFAULT_SIZE//2)), fg='blue')
                time.sleep(1)
                labels['front'].config(font=(DEFAULT_TYPE, DEFAULT_SIZE), fg='black')
                labels['right'].config(font=(DEFAULT_TYPE, DEFAULT_SIZE), fg='black')
                time.sleep(1.5)
            
            elif prediction.lower() == 'right_left':
                labels['front'].config(font=(DEFAULT_TYPE, DEFAULT_SIZE + (DEFAULT_SIZE//2)), fg='blue')
                labels['right'].config(font=(DEFAULT_TYPE, DEFAULT_SIZE + (DEFAULT_SIZE//2)), fg='blue')
                time.sleep(1)
                labels['front'].config(font=(DEFAULT_TYPE, DEFAULT_SIZE), fg='black')
                labels['right'].config(font=(DEFAULT_TYPE, DEFAULT_SIZE), fg='black')
                time.sleep(1.5)
            
            elif prediction.lower() == 'right_right':
                labels['right'].config(font=(DEFAULT_TYPE, DEFAULT_SIZE + (DEFAULT_SIZE//2)), fg='blue')
                labels['back'].config(font=(DEFAULT_TYPE, DEFAULT_SIZE + (DEFAULT_SIZE//2)), fg='blue')
                time.sleep(1)
                labels['right'].config(font=(DEFAULT_TYPE, DEFAULT_SIZE), fg='black')
                labels['back'].config(font=(DEFAULT_TYPE, DEFAULT_SIZE), fg='black')
                time.sleep(1.5)
            
            elif prediction.lower() == 'back_right':
                labels['right'].config(font=(DEFAULT_TYPE, DEFAULT_SIZE + (DEFAULT_SIZE//2)), fg='blue')
                labels['back'].config(font=(DEFAULT_TYPE, DEFAULT_SIZE + (DEFAULT_SIZE//2)), fg='blue')
                time.sleep(1)
                labels['right'].config(font=(DEFAULT_TYPE, DEFAULT_SIZE), fg='black')
                labels['back'].config(font=(DEFAULT_TYPE, DEFAULT_SIZE), fg='black')
                time.sleep(1.5)
            
            elif prediction.lower() == 'left_left':
                labels['left'].config(font=(DEFAULT_TYPE, DEFAULT_SIZE + (DEFAULT_SIZE//2)), fg='blue')
                labels['back'].config(font=(DEFAULT_TYPE, DEFAULT_SIZE + (DEFAULT_SIZE//2)), fg='blue')
                time.sleep(1)
                labels['left'].config(font=(DEFAULT_TYPE, DEFAULT_SIZE), fg='black')
                labels['back'].config(font=(DEFAULT_TYPE, DEFAULT_SIZE), fg='black')
                time.sleep(1.5)
            
            elif prediction.lower() == 'back_left':
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