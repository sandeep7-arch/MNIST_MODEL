import kagglehub
import pandas as pd
import torch
import torch.nn as nn
from main import mnist
path = kagglehub.dataset_download("oddrationale/mnist-in-csv")

df = pd.read_csv(f"{path}/mnist_train.csv")
df2 = pd.read_csv(f"{path}/mnist_test.csv")
train_y = df['label']
train_x = df.drop(columns=['label'])
X_tensor = torch.tensor(train_x.values , dtype=torch.float32)/255
Y_tensor = torch.tensor(train_y.values , dtype=torch.long)

model = mnist()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters() , lr=0.001)

epochs = 100
losses = []
for i in range(epochs):
    y_pred = model.forward(X_tensor)
    loss = criterion(y_pred , Y_tensor)
    losses.append(loss)
    if i%10==0:
        print(f'Epoch: {i} and loss = {loss}')
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
