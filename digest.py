import librosa
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.cluster import KMeans
from collections import defaultdict
from collections import Counter

class digest:
    def __init__(self):
        self.features = None
        self.labels = None

    # GAN Setup Care
    def build_generator(self):
        generator = Sequential([
            Dense(256, input_dim=100, activation='relu'),
            Dense(512, activation='relu'),
            Dense(256, activation='relu'),
            Dense(5, activation='sigmoid'),
            Reshape((5, 1))
        ])
        return generator

    # Noise Generator
    def generate_data(self, generator, num_samples):
        noise = np.random.normal(0, 1, size=(num_samples, 100))
        generated_data = generator.predict(noise)
        return generated_data

    # Dataset Creator this method is fairly large hower is in charge of obtaining our collected data and generating data used in the project.
    # It contains parameters such as the ability to augement, and generate data with various epochs to better the generated data.
    def data_set(self, augment=False, use_gan=True, gan_epochs=10, gan_batch_size=32):
        folders = ["MAsite", "MBsite", "MMID", "MTsite"]
        flat_features = []
        tags = []
        class_samples = defaultdict(list)
        max_samples = 0

        for d in folders:
            for file in os.listdir("./"+d):
                y, sr = librosa.load("./"+d+"/"+file)

                if augment:
                    y_stretch = librosa.effects.time_stretch(y, rate=1.2)
                    y = np.pad(y_stretch, (0, len(y) - len(y_stretch)), mode='constant')

                fft_features = np.abs(librosa.stft(y))

                sample = np.mean(fft_features, axis=1)
                sample = np.array_split(sample, 205)
                
                class_label = file.split(".")[0]
                class_label = "_".join(word.capitalize() for word in class_label.split("_")[1:])
                class_samples[class_label].extend(sample)

                if len(class_samples[class_label]) > max_samples:
                    max_samples = len(class_samples[class_label])

        for label, samples in class_samples.items():
            oversampled_samples = samples * (max_samples // len(samples))
            remaining_samples = max_samples % len(samples)
            oversampled_samples += samples[:remaining_samples]
            flat_features += oversampled_samples
            tags += [label] * len(oversampled_samples)

        if use_gan:
            X_gan = np.array(flat_features)
            X_gan = np.reshape(X_gan, (X_gan.shape[0], X_gan.shape[1], 1))
            
            generator = self.build_generator()
            generator.compile(loss='binary_crossentropy', optimizer='adam')
            generator.fit(np.random.normal(size=(X_gan.shape[0], 100)), X_gan, epochs=gan_epochs, batch_size=gan_batch_size, verbose=0)
            
            min_synthetic_samples = 100000
            num_synthetic_samples = max(max_samples - len(class_samples[max(class_samples, key=len)]), min_synthetic_samples)

            if num_synthetic_samples > 0:
                synthetic_data = []

                batch_size = 100  
                num_batches = (num_synthetic_samples + batch_size - 1) // batch_size

                for _ in range(num_batches):
                    batch_synthetic_data = self.generate_data(generator, batch_size)
                    synthetic_data.extend(batch_synthetic_data)

                synthetic_data = np.array(synthetic_data)[:num_synthetic_samples] 
                synthetic_data = np.reshape(synthetic_data, (num_synthetic_samples, 5))

                flat_features = np.array(flat_features) 
                if flat_features.shape[1] != synthetic_data.shape[1]:
                    print("Error: Shapes of flat_features and synthetic_data don't match.")
                else:
                    X_gan = np.array(flat_features)
                    X_gan = np.reshape(X_gan, (X_gan.shape[0], X_gan.shape[1], 1))
                    
                    generator = self.build_generator()
                    generator.compile(loss='binary_crossentropy', optimizer='adam')
                    generator.fit(np.random.normal(size=(X_gan.shape[0], 100)), X_gan, epochs=gan_epochs, batch_size=gan_batch_size, verbose=0)
                    
                    synthetic_data = []
                    max_samples_per_class = 10000
                    for label, samples in class_samples.items():
                        num_synthetic_samples = max_samples_per_class - len(samples)
                        if num_synthetic_samples > 0:
                            batch_size = 100
                            num_batches = (num_synthetic_samples + batch_size - 1) // batch_size

                            for _ in range(num_batches):
                                batch_synthetic_data = self.generate_data(generator, batch_size)
                                synthetic_data.extend(batch_synthetic_data)

                    synthetic_data = np.array(synthetic_data)
                    synthetic_data = synthetic_data.reshape((synthetic_data.shape[0], synthetic_data.shape[1]))
                    flat_features = np.concatenate((flat_features, synthetic_data), axis=0)

                    kmeans = KMeans(n_clusters=len(class_samples))
                    kmeans.fit(synthetic_data)
                    synthetic_cluster_labels = kmeans.predict(synthetic_data)
                    new_labels = [list(set(tags))[label] for label in synthetic_cluster_labels]
                    tags += new_labels
    
        self.features = flat_features
        self.labels = tags

        return tags, flat_features
    
    # This is used when obtaing and audio file to create a uniform way to format individual files.
    def extract_features(self,directory, file):
        if directory == "./":
            y, sr = librosa.load("./"+file)
        else:
            y, sr = librosa.load("./"+directory+"/"+file)

        fft_features = np.abs(librosa.stft(y))

        sample = np.mean(fft_features, axis=1)
        sample = np.array_split(sample, 205)
        return sample[0]
            