import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
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

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)



# Feature normalization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize KNN classifier
knn = KNeighborsClassifier(n_neighbors=16)  # Adjust k as needed

# Train the classifier
knn.fit(X_train_scaled, y_train)

joblib.dump(knn, 'knn_model.pkl')
# Predict on test data
y_pred = knn.predict(X_test_scaled)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


folder = 'MAsite'
new_audio_file = 'MA_front.wav' # Path to new audio file
print("Running Model On: " + folder + "," + new_audio_file)
new_features = dataProcess.extract_features(folder,new_audio_file)
prediction = knn.predict([new_features])
print("Predicted direction:", prediction[0])

