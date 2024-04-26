import numpy as np
import librosa
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from digest import digest
from sklearn.preprocessing import LabelEncoder

print("Beginning Data Processing")
dataProcess = digest()
y, x = dataProcess.data_set()
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

# Encode categorical labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Train an SVM model
print("Training SVM model")
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_model.fit(X_train, y_train_encoded)
print("SVM model training complete.")

# Generate SVM predictions
svm_train_preds = svm_model.predict(X_train)
svm_test_preds = svm_model.predict(X_test)

# Ensure svm_train_preds and svm_test_preds have compatible shapes with X_train and X_test
svm_train_preds = svm_train_preds.reshape(-1, 1)
svm_test_preds = svm_test_preds.reshape(-1, 1)

# Check shapes
print("Shapes before concatenation:")
print("X_train shape:", X_train.shape)
print("svm_train_preds shape:", svm_train_preds.shape)
print("X_test shape:", X_test.shape)
print("svm_test_preds shape:", svm_test_preds.shape)


# Concatenate SVM predictions with original features
X_train_stacked = np.hstack((X_train, svm_train_preds))
X_test_stacked = np.hstack((X_test, svm_test_preds))

# Train a Random Forest model using stacked features
print("Training Random Forest model with stacked features")
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train_stacked, y_train_encoded)
print("Random Forest model training complete.")

# Evaluate the Random Forest model
print("Evaluating Random Forest model")
y_pred = rf_model.predict(X_test_stacked)
accuracy = accuracy_score(y_test_encoded, y_pred)
print("Evaluation complete.")
print("Accuracy:", accuracy)

# Prediction on new audio
'''
folder = 'MAsite'
new_audio_file = 'MA_front_left.wav'  # Path to new audio file
print("Running Model On: " + folder + "," + new_audio_file)
new_features = dataProcess.extract_features(folder, new_audio_file)
svm_prediction = svm_model.predict([new_features])
# Reshape SVM prediction to match dimensions for stacking
svm_prediction = svm_prediction.reshape(-1, 1)
stacked_features = np.hstack((new_features, svm_prediction))
rf_prediction = rf_model.predict([stacked_features])
print("Predicted direction:", rf_prediction[0])'''
