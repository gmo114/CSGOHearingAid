import numpy as np
import librosa
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
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
model = RandomForestClassifier(n_estimators=13)
model.fit(X_train, y_train)
joblib.dump(model, 'RandomForest_model.pkl')
print("Model training complete.")

# Model evaluation
print("Evaluating")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)
print("Evaluation complete.")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("Confusion Matrix:\n", conf_matrix)

# Prediction on new audio
folder = 'MMID'
new_audio_file = 'MM_Right.wav' # Path to new audio file
print("Running Model On: " + folder + "," + new_audio_file)
new_features = dataProcess.extract_features(folder,new_audio_file)
prediction = model.predict([new_features])
print("Predicted direction:", prediction[0])