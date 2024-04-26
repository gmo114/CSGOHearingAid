import librosa
import os
import numpy as np


def data_set():
    # Load audio file
    folders = ["MAsite","MBsite","MMID","MTsite"]
    flat_features = []
    tags = []
    for d in folders:
        for file in os.listdir("./"+d):
            y, sr = librosa.load("./"+d+"/"+file)
            # Extract features using FFT
            fft_features = np.abs(librosa.stft(y))
            # Flatten the features to create a feature vector
            sample = np.mean(fft_features, axis=1)
        
            sample = np.array_split(sample, 205)
            flat_features += sample
           
            for  a in sample:
                print(len(a))
                tags.append(file.split(".")[0])
    
    return tags,flat_features