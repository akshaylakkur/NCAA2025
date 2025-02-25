'nn.BatchNorm1d(20, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),'
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset, TensorDataset
import pandas as pd
from torchvision.ops import MLP

class Prediction(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Prediction, self).__init__()
        # self.historic_data = historic_data
        # self.embedding = nn.Embedding(2026, 8)
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

model = Prediction(35, 256, 35)
data = pd.DataFrame(pd.read_csv('male_data.csv'))
data = torch.from_numpy(data.to_numpy()).float()
data = TensorDataset(data)
dataloader = DataLoader(data, batch_size=128, shuffle=False)

'training loop'
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()
epoch = 100

for epoch in range(epoch):
    model.train()
    for item in dataloader:
        optimizer.zero_grad()
        outputs = model(item[0])
        loss = loss_fn(outputs, item[0])
        loss.backward()
        optimizer.step()
        print(loss.item())

torch.save(model.state_dict(), 'model_weights.pth')