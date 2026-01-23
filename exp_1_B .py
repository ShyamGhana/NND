# exp-1_B 

from google.colab import files
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

# =========================
# 1. Load MNIST Dataset
# =========================
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

# =========================
# 2. Build Model (same as yours)
# =========================
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3)

# =========================
# 3. Upload Two-Digit Image
# =========================
print("Upload your two-digit image (example: 45, 78, 23)")
uploaded = files.upload()
image_name = list(uploaded.keys())[0]

# =========================
# 4. Preprocess Image
# =========================
img = Image.open(image_name).convert("L")

# Resize image to width = 56, height = 28 (two digits side by side)
img = img.resize((56, 28))

img_array = np.array(img)
img_array = 255 - img_array   # invert colors
img_array = img_array / 255.0

# =========================
# 5. Split Image into Two Digits
# =========================
digit1 = img_array[:, :28]   # left half
digit2 = img_array[:, 28:]   # right half

digit1 = digit1.reshape(1, 28, 28)
digit2 = digit2.reshape(1, 28, 28)

# =========================
# 6. Predict Each Digit
# =========================
pred1 = np.argmax(model.predict(digit1))
pred2 = np.argmax(model.predict(digit2))

# =========================
# 7. Show Results
# =========================
plt.figure(figsize=(6, 3))

plt.subplot(1, 2, 1)
plt.imshow(digit1.reshape(28, 28), cmap="gray")
plt.title(f"Digit 1: {pred1}")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(digit2.reshape(28, 28), cmap="gray")
plt.title(f"Digit 2: {pred2}")
plt.axis("off")

plt.show()

print("Predicted Two-Digit Number:", str(pred1) + str(pred2))

