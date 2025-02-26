'nn.BatchNorm1d(20, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),'
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset, TensorDataset
import pandas as pd
from torchvision.ops import MLP

torch.manual_seed(1)
data = pd.read_csv('male_data.csv')
wteam, lteam = data['WTeamID'], data['LTeamID']
#data = data.drop(columns=['WTeamID', 'LTeamID'])
wteam, lteam, data = torch.from_numpy(wteam.to_numpy()).long(), torch.from_numpy(lteam.to_numpy()).long(), torch.from_numpy(data.to_numpy()).float()
swap = torch.randn(len(wteam)) > 0.5
t1 = torch.where(swap, lteam,wteam)
t2 = torch.where(swap, wteam,lteam)
wins = torch.ones_like(t1, dtype=torch.float32)
wins[swap] = 0
wins = wins.unsqueeze(1)

class Prediction(nn.Module):
    def __init__(self, training_input=35, hidden_dim=256, embedding_dim=8, output_dim=1, num_teams=1481):
        super(Prediction, self).__init__()
        self.team_embedding = nn.Embedding(num_teams, embedding_dim)
        self.lstm = nn.LSTM(training_input, hidden_dim, batch_first=True)
        self.emb_to_hidden = nn.Linear(2 * embedding_dim, hidden_dim)
        self.training_fc1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
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

        #Input -> [2,16]
        self.testing_fc1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim,hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
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
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.fc3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.fc4 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.prediction_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.Linear(hidden_dim*2, hidden_dim*4),
            nn.Linear(hidden_dim*4, hidden_dim*8),
            nn.Linear(hidden_dim*8, output_dim),
            nn.Sigmoid()
        )
    def forward(self, team1, team2, features=None, training=True):
        self.team1_emb = self.team_embedding(team1)
        self.team2_emb = self.team_embedding(team2)
        tms = torch.cat([self.team1_emb, self.team2_emb], dim=1)
        x = self.emb_to_hidden(tms)

        if training:
            self.lstm_out, (self.h_n, self.c_n) = self.lstm(features.unsqueeze(1))
            self.lstm_out = self.lstm_out.squeeze(1)
            x = x + self.lstm_out
            x = self.training_fc1(x)
        else:
            x = self.testing_fc1(x)

        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.prediction_layer(x)
        return x

model = Prediction()
tensor_dataset = TensorDataset(t1, t2, data)
dataloader = DataLoader(tensor_dataset, batch_size=128, shuffle=False)
epochs = 100

for t1, t2, features in dataloader:
    x = model(t1, t2, training=False)
    break