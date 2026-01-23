from google.colab import files
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

# Load MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

# Model
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3)

# Upload image
print("Upload your handwritten digit image")
uploaded = files.upload()
image_name = list(uploaded.keys())[0]

# Preprocess image
img = Image.open(image_name).convert("L")
img = img.resize((28, 28))
img_array = np.array(img)
img_array = 255 - img_array
img_array = img_array / 255.0
img_array = img_array.reshape(1, 28, 28)

# Predict
prediction = model.predict(img_array)
digit = np.argmax(prediction)

# Show result
plt.imshow(img_array.reshape(28, 28), cmap="gray")
plt.title(f"Predicted Digit: {digit}")
plt.axis("off")
plt.show()

print("Predicted digit is:", digit)
