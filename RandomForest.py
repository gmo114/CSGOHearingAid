import numpy as np
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from digest import digest

print("Beginning Data Processing")
dataProcess = digest()
y,x = dataProcess.data_set()
x = np.array(x)
y = np.array(y)
print("Data processing complete.")
print("X shape:", x.shape)
print("y shape:", y.shape)

# Split data into train and test sets
print("Splitting Data")
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
print("Data split complete.")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Model training
print("Training")
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
print("Model training complete.")

# Model evaluation
print("Evaluating")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Evaluation complete.")
print("Accuracy:", accuracy)

# Prediction on new audio
folder = 'MAsite'
new_audio_file = 'MA_front.wav' # Path to new audio file
print("Running Model On: " + folder + "," + new_audio_file)
new_features = dataProcess.extract_features(folder,new_audio_file)
prediction = model.predict([new_features])
print("Predicted direction:", prediction[0])