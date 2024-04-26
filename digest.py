import librosa
import os
import numpy as np

class digest:
    def __init__(self):
        self.features = None
        self.labels = None


    def data_set(self):
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
            
                for a in sample:
                    tags.append(file.split(".")[0])
        self.features = flat_features
        self.labels = tags
        return tags,flat_features
    
    def extract_features(self,directory, file):
        y, sr = librosa.load("./"+directory+"/"+file)
        # Extract features using FFT
        fft_features = np.abs(librosa.stft(y))
        # Flatten the features to create a feature vector
        sample = np.mean(fft_features, axis=1)
        sample = np.array_split(sample, 205)
        return sample[0]
            