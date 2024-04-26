import numpy as np
import os
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score



def extract_features():
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


lables,flat_features = extract_features()
print(len(lables))
print("\n")
print(len(flat_features))