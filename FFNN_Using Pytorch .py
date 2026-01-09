# implement free forward NN using Pytorch
Date : 09/01/2026
Day : Friday 

import torch
import torch.nn as nn

x = torch.tensor([[1., 2., 3.]])
y = torch.tensor([[1.]])

model = nn.Sequential(
    nn.Linear(3, 2),
    nn.Sigmoid(),
    nn.Linear(2, 1),
    nn.Sigmoid()
)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for _ in range(100):
    y_pred = model(x)
    loss = criterion(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("Prediction:", model(x).detach().numpy())

# OUTPUT 

Prediction: [[0.8748543]]

