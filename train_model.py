# -*- coding: utf-8 -*-

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import csv
import numpy as np
import matplotlib.pyplot as plt
import time

normalization = 94855
device = torch.device("cpu")

def s_to_hrmins(s):
    hr = s//(60**2)
    s -= hr*(60**2)
    minutes = s//60
    s -= 60 * minutes
    return f"{hr}hr {minutes}min {s}s"

with open("data/tw_data_v2.csv", newline='') as f:
    train = list(csv.reader(f))
    train = np.array(train).astype(np.int64).flatten()
    
    train = train / normalization
    size = 30
    train = np.lib.stride_tricks.sliding_window_view(train, size)
    print(train.shape)
    train = train.reshape((train.shape[0], 1, size))
    train = torch.tensor(train)
    train = train.to(device)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(30, 23)
        self.fc2 = nn.Linear(23, 15)
        self.fc3 = nn.Linear(15, 8)
        self.fc4 = nn.Linear(8, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

model = Net()
model.to(device)

losses = []
loss_average = 0
set_loss = nn.MSELoss()
optimizer = optim.NAdam(model.parameters(), lr = 2e-5)

data_size = train.shape[0]
idx = [i for i in range(data_size - 1)]


batch = 6000

print(train.shape[0])

time_a = time.time()

for epoch in range(batch):
    random.shuffle(idx)
    loss_average = 0

    for index in range(data_size - 1):  
        optimizer.zero_grad()
        
        data = train[idx[index]].float()
        label = train[idx[index] + 1][0][-1].reshape(1, -1).float()

        out = model(data)

        loss = set_loss(out, label)

        loss_average += loss.item()

        loss.backward()
        optimizer.step()

        use_time = time.time() - time_a

        remaining_time = int(use_time / ((index + 1) + epoch * data_size) * ((data_size * batch) - (index + epoch * data_size)))
        remaining_time = s_to_hrmins(remaining_time)
    
    loss_average = loss_average / data_size
    losses.append(loss_average)

    print(f"{int((epoch + 1) / batch * 100) }%, {epoch + 1}, Remaining time: {remaining_time}, loss average: {loss_average / data_size}")

torch.save(model.state_dict(), 'model_data.pt')

plt.plot(losses)
plt.show()