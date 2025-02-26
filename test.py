import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

files = '/Users/akshaylakkur/PycharmProjects/MarchMadness/march-machine-learning-mania-2025/SampleSubmissionStage2.csv'
matchups = pd.read_csv(files)
ids = matchups['ID']
ids = ids.str.split('_', expand=True)
season, team1, team2 = pd.to_numeric(ids[0]), pd.to_numeric(ids[1]), pd.to_numeric(ids[2])

team1, team2 = torch.from_numpy(team1.to_numpy()).long(), torch.from_numpy(team2.to_numpy()).long()

dataset = TensorDataset(team1, team2)
loader = DataLoader(dataset, batch_size=256, shuffle=False)