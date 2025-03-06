import pandas as pd
import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
import os
from torchvision.ops import MLP
import torch.nn.functional as F

csv = '/Users/akshaylakkur/PycharmProjects/MarchMadness/current/files/TrainingData.csv'
data = pd.read_csv(csv)
files = os.listdir('/Users/akshaylakkur/PycharmProjects/MarchMadness/current/indiv_team_data')

def swap(data):
    wteam, lteam = data['WTeamID'], data['LTeamID']
    wteam, lteam, data = torch.from_numpy(wteam.to_numpy()).long(), torch.from_numpy(lteam.to_numpy()).long(), torch.from_numpy(data.to_numpy()).float()
    swap = torch.randn(len(wteam)) > 0.5
    team1 = torch.where(swap, lteam,wteam)
    team2 = torch.where(swap, wteam,lteam)
    wins = torch.ones_like(team1, dtype=torch.float32)
    wins[swap] = 0
    wins = wins.unsqueeze(1)
    return wins

def test_inputs(team1:int, team2:int):
    t1 = f'/Users/akshaylakkur/PycharmProjects/MarchMadness/current/indiv_team_data/historic_data_{team1}.csv'
    t2 = f'/Users/akshaylakkur/PycharmProjects/MarchMadness/current/indiv_team_data/historic_data_{team2}.csv'
    team_data_1 = pd.read_csv(t1)
    team_data_2 = pd.read_csv(t2)
    t1 = torch.from_numpy(team_data_1.to_numpy()).float()
    t2 = torch.from_numpy(team_data_2.to_numpy()).float()
    return t1, t2, team_data_1, team_data_2


backbone = MLP(36, [64, 128, 256, 256, 256, 256, 256], norm_layer=nn.LayerNorm, activation_layer=nn.ReLU, inplace=True, bias=True, dropout=0.2)
output_perceptron = MLP(256, [256, 128, 128, 64], norm_layer=nn.LayerNorm, activation_layer=nn.ReLU, bias=True, dropout=0.2)
class TransformerModel(nn.Module):
    def __init__(self, MLPBackbone, outputPerceptron):
        super(TransformerModel, self).__init__()
        self.backbone = MLPBackbone
        transformer_encoder_layer = nn.TransformerEncoderLayer(256, 4, batch_first=True)
        self.encoder = nn.TransformerEncoder(transformer_encoder_layer, num_layers=3)
        self.output_perceptron = outputPerceptron
        self.prediction_setup = nn.Linear(64,1)
    def forward(self, t1, t2, training=True):
        if training:
            mem = torch.cat((t1, t2), dim=0)
            x = self.backbone(mem)
            x = self.encoder(x)
            x = self.output_perceptron(x)
            x = self.prediction_setup(x)
        else:
            mem = torch.cat((t1, t2), dim=0)
            x = self.backbone(mem)
            x = self.encoder(x)
            x = self.output_perceptron(x)
            x = self.prediction_setup(x).view(-1)
            pred_layer = nn.Linear(x.shape[0], 1)
            x = pred_layer(x)
        return torch.sigmoid(x)


model = TransformerModel(backbone, output_perceptron)
t1, t2, raw1, raw2 = test_inputs(1245, 1480)
wins1 = swap(raw1)
wins2 = swap(raw2)
wins = torch.cat((wins1, wins2), dim=0)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
model.train()
optimizer.zero_grad()
out = model(t1, t2, training=False)
print(out)