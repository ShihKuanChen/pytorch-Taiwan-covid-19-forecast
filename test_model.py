print("Loading Data...")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

normalization = 94855

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
model.load_state_dict(torch.load("model_data.pt"))

model.eval()

print("Loading Done!")

def inputs(x):
    data = []
    for i in range(x):
        data.append(int(input(f"Day {i + 1}: ")))
    return np.array([data])

input_data = inputs(30) / normalization
    
input_data = torch.tensor(input_data).float()

out = model(input_data)
out_a = out * normalization

print(f"預計確診人數: {int(out_a[0][0])}人")
