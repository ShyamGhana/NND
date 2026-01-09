#implement free forward NN using tensorflow and keras
Date : 09/01/2026
Day : Friday 

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x = np.array([[1,2,3]], dtype=float)
y = np.array([[1]], dtype=float)

model = Sequential([Dense(2, activation="sigmoid", input_shape=(3,)),
                    Dense(1, activation="sigmoid")])

model.compile(optimizer="sgd", loss="mse")
model.fit(x, y, epochs=100, verbose=0)
print("Prediction:", model.predict(x))


# OUTPUT 

1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 66ms/step
Prediction: [[0.6017688]]
