import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from torchvision.ops import MLP

data = pd.DataFrame(pd.read_csv('male_data.csv'))
data = torch.from_numpy(data.to_numpy()).float()

class KnowledgeNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(KnowledgeNet, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim, affine=True, track_running_stats=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim, affine=True, track_running_stats=True),
            nn.Dropout(0.2),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim*4),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim, affine=True, track_running_stats=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim*4, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim, affine=True, track_running_stats=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.learn_weights = nn.Linear(hidden_dim, input_dim)
    def forward(self, data):
        data = self.fc1(data)
        data = self.fc2(data)
        data = self.learn_weights(data)
        return data

mlp = MLP(35, [10,20,30,40,50,60,70,60,50,40,30,20,35], nn.BatchNorm1d, nn.ReLU, True, True, 0.3)

'Train data on itself'
class Prediction(nn.Module):
    def __init__(self,  input_dim, hidden_dim, out_dim):
        super(Prediction, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(int(input_dim), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, out_dim),
            nn.Sigmoid()
        )

    def forward(self, team1, team2):
        x = torch.cat((team1, team2), dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


model = Prediction(2, 3, 1)
t1 = torch.tensor([1101.0]).unsqueeze(0)
t2 = torch.tensor([1201.0]).unsqueeze(0)
t = model(t1,t2)
print(t)
