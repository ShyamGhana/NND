# Exp_2 : Back propogation for model to learn pattern of fraud detection & normal activity.

# import dataset 

import os, pandas as pd, numpy as np

file_path = "/content/creditcard.csv"

if os.path.exists(file_path):
    os.remove(file_path)

print("Downloading dataset...")


!curl -L -o /content/creditcard.csv https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv


file_size = os.path.getsize(file_path)
print("File size:", file_size, "bytes")

if file_size == 0:
    raise Exception("Dataset download failed! File is empty.")


data = pd.read_csv(file_path)
print("\nDataset loaded successfully!")
print("Shape:", data.shape)
print(data.head())


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

X = data.drop("Class", axis=1)
y = data["Class"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

model = Sequential([
    Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("\nTraining model...")
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print("\nTest Accuracy:", accuracy)

sample = X_test[0].reshape(1, -1)
prediction = model.predict(sample)[0][0]

print("\nPrediction Probability:", prediction)
print("Fraud" if prediction > 0.5 else "Normal Transaction")
