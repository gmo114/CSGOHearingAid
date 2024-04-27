import librosa
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape, Flatten
from tensorflow.keras.optimizers import Adam
from collections import defaultdict

class digest:
    def __init__(self):
        self.features = None
        self.labels = None


    def build_generator(self):
        generator = Sequential([
            Dense(256, input_dim=100, activation='relu'),
            Dense(512, activation='relu'),
            Dense(256, activation='relu'),
            Dense(5, activation='sigmoid'),  # Adjust output dimensions to match your feature size
            Reshape((5, 1))
        ])
        return generator

    def generate_data(self, generator, num_samples):
        noise = np.random.normal(0, 1, size=(num_samples, 100))
        generated_data = generator.predict(noise)
        return generated_data

    def data_set(self, augment=True, use_gan=True, gan_epochs=10, gan_batch_size=32):
        folders = ["MAsite", "MBsite", "MMID", "MTsite"]
        flat_features = []
        tags = []
        class_samples = defaultdict(list)
        max_samples = 0

        # Collect samples per class
        for d in folders:
            for file in os.listdir("./"+d):
                y, sr = librosa.load("./"+d+"/"+file)

                # Augment data if specified
                if augment:
                    # Example augmentation: time stretching
                    y_stretch = librosa.effects.time_stretch(y, rate=1.2)
                    y = np.pad(y_stretch, (0, len(y) - len(y_stretch)), mode='constant')

                # Extract features using FFT
                fft_features = np.abs(librosa.stft(y))

                # Flatten the features to create a feature vector
                sample = np.mean(fft_features, axis=1)
                sample = np.array_split(sample, 205)
                
                # Add samples to respective class lists
                class_label = file.split(".")[0]
                class_label = "_".join(class_label.split("_")[1:])
                class_samples[class_label].extend(sample)

                if len(class_samples[class_label]) > max_samples:
                    max_samples = len(class_samples[class_label])

        # Balance classes by oversampling
        for label, samples in class_samples.items():
            oversampled_samples = samples * (max_samples // len(samples))
            remaining_samples = max_samples % len(samples)
            oversampled_samples += samples[:remaining_samples]
            flat_features += oversampled_samples
            newL = "_".join(label.split("_")[1:])
            tags += [newL] * len(oversampled_samples)

        # If GAN is enabled, generate additional synthetic data
        if use_gan:
            # Reshape data for GAN training
            X_gan = np.array(flat_features)
            X_gan = np.reshape(X_gan, (X_gan.shape[0], X_gan.shape[1], 1))
            
            # Build and train GAN
            generator = self.build_generator()
            generator.compile(loss='binary_crossentropy', optimizer='adam')
            generator.fit(np.random.normal(size=(X_gan.shape[0], 100)), X_gan, epochs=gan_epochs, batch_size=gan_batch_size, verbose=0)
            
            # Generate synthetic data
            # Ensure a minimum number of synthetic samples is generated
            min_synthetic_samples = 100000  # Adjust as needed
            num_synthetic_samples = max(max_samples - len(class_samples[max(class_samples, key=len)]), min_synthetic_samples)

            # Generate synthetic data only if num_synthetic_samples > 0
            if num_synthetic_samples > 0:
                synthetic_data = []

    # Generate synthetic data in batches
                batch_size = 100  # Adjust batch size as needed
                num_batches = (num_synthetic_samples + batch_size - 1) // batch_size

                for _ in range(num_batches):
                    batch_synthetic_data = self.generate_data(generator, batch_size)
                    synthetic_data.extend(batch_synthetic_data)

                # Reshape synthetic data to match shape of real data
                synthetic_data = np.array(synthetic_data)[:num_synthetic_samples]  # Trim excess samples
                synthetic_data = np.reshape(synthetic_data, (num_synthetic_samples, 5))

                flat_features += synthetic_data.tolist()
                newL = "_".join(label.split("_")[1:])

                tags += [newL] * num_synthetic_samples
                
        # Balance classes by oversampling
        for label, samples in class_samples.items():
            oversampled_samples = samples * (max_samples // len(samples))
            remaining_samples = max_samples % len(samples)
            oversampled_samples += samples[:remaining_samples]
            flat_features += oversampled_samples
            newL = "_".join(label.split("_")[1:])
            tags += [newL] * len(oversampled_samples)

            # Check lengths after appending synthetic data
        print(len(tags), len(flat_features))
        self.features = flat_features
        self.labels = tags

        return tags, flat_features
    
    def extract_features(self,directory, file):
        if directory == "./":
            y, sr = librosa.load("./"+file)
        else:
            y, sr = librosa.load("./"+directory+"/"+file)
            
        # Extract features using FFT
        fft_features = np.abs(librosa.stft(y))
        # Flatten the features to create a feature vector
        sample = np.mean(fft_features, axis=1)
        sample = np.array_split(sample, 205)
        return sample[0]
            