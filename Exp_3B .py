
# Exp_2: Feed Forward Neural Network for Fraud Detection
# With Behavioral Data


# Step 1: Import libraries
import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Step 2: Download Dataset
file_path = "/content/creditcard.csv"

if os.path.exists(file_path):
    os.remove(file_path)

print("Downloading dataset...")

!curl -L -o /content/creditcard.csv https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv

file_size = os.path.getsize(file_path)
print("File size:", file_size, "bytes")

if file_size == 0:
    raise Exception("Dataset download failed! File is empty.")

# Step 3: Load Dataset
data = pd.read_csv(file_path)
print("\nDataset loaded successfully!")
print("Shape:", data.shape)
print(data.head())

# Step 4: Add Behavioral Data Features
# Behavioral features are created from existing transaction data

# Transaction frequency behavior (rolling count)
data['Transaction_Frequency'] = data.groupby('Class').cumcount()

# Average transaction amount behavior
data['Avg_Amount'] = data['Amount'].rolling(window=10, min_periods=1).mean()

# Time-based behavior (normalized time feature)
data['Time_Normalized'] = (data['Time'] - data['Time'].mean()) / data['Time'].std()

# Amount deviation behavior
data['Amount_Deviation'] = data['Amount'] - data['Amount'].mean()

print("\nBehavioral features added successfully!")
print(data[['Transaction_Frequency', 'Avg_Amount', 'Time_Normalized', 'Amount_Deviation']].head())

# Step 5: Feature Selection
X = data.drop("Class", axis=1)
y = data["Class"]

# Step 6: Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 7: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Step 8: Build Feed Forward Neural Network Model
model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Step 9: Compile Model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Step 10: Train Model
print("\nTraining model...")
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2, verbose=1)

# Step 11: Evaluate Model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print("\nTest Accuracy:", accuracy)

# Step 12: Prediction on Sample Data
sample = X_test[0].reshape(1, -1)
prediction = model.predict(sample)[0][0]

print("\nPrediction Probability:", prediction)
print("Fraud Transaction" if prediction > 0.5 else "Normal Transaction")
