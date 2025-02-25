'nn.BatchNorm1d(20, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),'
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset, TensorDataset
import pandas as pd
from torchvision.ops import MLP


data = pd.read_csv('male_data.csv')

season, wteam, lteam = data['Season'], data['WTeamID'], data['LTeamID']
data = data.drop(columns=['Season', 'WTeamID', 'LTeamID'])
season, wteam, lteam, data = torch.from_numpy(season.to_numpy()).long(), torch.from_numpy(wteam.to_numpy()).long(), torch.from_numpy(lteam.to_numpy()).long(), torch.from_numpy(data.to_numpy()).float()
swap = torch.randn(len(wteam)) > 0.5
t1 = torch.where(swap, lteam,wteam)
t2 = torch.where(swap, wteam,lteam)
wins = torch.ones_like(t1)
wins[swap] = 0

class MemoryNet(nn.Module):
    def __init__(self, num_teams, num_seasons, input_dim, hidden_dim=256, embedding_dim=8, output_dim=1):
        super(MemoryNet, self).__init__()

        self.team_embedding = nn.Embedding(num_teams, embedding_dim)
        self.season_embedding = nn.Embedding(num_seasons, embedding_dim)

        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
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
        )

        self.inference_layer = nn.Sequential(
            nn.Linear(2 * embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.combined_fc = nn.Sequential(
            nn.Linear(2 * embedding_dim + hidden_dim + 248, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )
    def forward(self, season=None, team1=None, team2=None, features=None, infer=False):
        if infer:
            team1_emb = self.team_embedding(team1)
            team2_emb = self.team_embedding(team2)
            season_emb = self.season_embedding(season)
            concat = torch.cat((team1_emb, team2_emb, season_emb), dim=1)
            concat = self.inference_layer(concat)
            concat = self.combined_fc(concat)
            return concat
        else:
            team1_emb = self.team_embedding(team1)
            team2_emb = self.team_embedding(team2)
            season_emb = self.season_embedding(season)

            teams = self.inference_layer(torch.cat((team1_emb, team2_emb), dim=1))

            features = self.fc1(features)
            features = self.fc2(features)
            features = self.fc3(features)

            concat = torch.cat((teams, season_emb, features), dim=1)
            out = self.combined_fc(concat)
            return out



model = MemoryNet(1481,2025,32)
loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

data = TensorDataset(season, t1, t2, data, wins)
dataloader = DataLoader(data, batch_size=128, shuffle=True)
epochs = 100

# for epoch in range(epochs):
model.train()
for season, team1, team2, features, winners in dataloader:
    optimizer.zero_grad()
    num = model(season, team1, team2, features)
    print(num)
    break