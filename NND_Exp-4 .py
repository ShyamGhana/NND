# ============================================================
# AI FACE ATTENDANCE SYSTEM
# Neural Networks & Deep Learning
# ============================================================

!pip install keras-facenet opencv-python pandas scikit-learn

import cv2
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from google.colab import files

# ---------------- TRAINING ----------------

print("Upload TRAINING images")
uploaded = files.upload()

images=[]
labels=[]

for filename in uploaded.keys():

    img=cv2.imread(filename)
    img=cv2.resize(img,(160,160))
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    images.append(img)

    # ---- SMART LABEL EXTRACTION ----
    name = filename.lower()

    if "elon" in name:
        labels.append("Elon_Musk")
    elif "sundar" in name:
        labels.append("Sundar_Pichai")
    elif "tim" in name:
        labels.append("Tim_Cook")
    else:
        labels.append("Student")

images=np.array(images)

# ---------------- FACENET ----------------

embedder=FaceNet()
embeddings=embedder.embeddings(images)

# ---------------- LABEL ENCODER ----------------

encoder=LabelEncoder()
labels_encoded=encoder.fit_transform(labels)

# ---------------- MODEL ----------------

model=Sequential([
    Dense(256,activation='relu',input_shape=(512,)),
    Dense(128,activation='relu'),
    Dense(len(set(labels_encoded)),activation='softmax')
])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(embeddings,labels_encoded,epochs=10)

# ---------------- ATTENDANCE ----------------

def mark_attendance(name):
    df=pd.DataFrame({"Name":[name],"Status":["Present"]})
    df.to_csv("attendance.csv",mode='a',header=False,index=False)
    print(name,"marked present!")

# ---------------- TEST ----------------

print("Upload test image")
uploaded=files.upload()

for filename in uploaded.keys():

    img=cv2.imread(filename)
    face=cv2.resize(img,(160,160))
    face=cv2.cvtColor(face,cv2.COLOR_BGR2RGB)

    embedding=embedder.embeddings([face])
    prediction=model.predict(embedding)

    confidence=np.max(prediction)
    label=np.argmax(prediction)
    name=encoder.inverse_transform([label])[0]

    print("Detected 1 face(s) in the image.")
    print("Predicted Name:",name)
    print("Prediction Confidence:",confidence)

    if confidence>0.6:
        mark_attendance(name)
    else:
        print("Unknown face")

    plt.imshow(face)
    plt.title(f"{name} (Conf: {confidence:.2f})")
    plt.axis("off")
    plt.show()
