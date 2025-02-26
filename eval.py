import torch
import torch.nn as nn
from torchvision.ops import MLP

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
            nn.Linear(3 * embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.combined_fc = nn.Sequential(
            nn.Linear(3 * embedding_dim + hidden_dim + 240, hidden_dim),
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
            inf = self.inference_layer(concat)
            #dummy_features = torch.zeros(1, 240, device=inf.device)

            # Concatenate all required inputs for combined_fc
            final_concat = torch.cat((team1_emb, team2_emb, season_emb, inf), dim=1)
            print(final_concat.shape)
            out = self.combined_fc(final_concat)
            return out
        else:
            team1_emb = self.team_embedding(team1)
            team2_emb = self.team_embedding(team2)
            season_emb = self.season_embedding(season)

            teams = self.inference_layer(torch.cat((team1_emb, team2_emb, season_emb), dim=1))

            features = self.fc1(features)
            features = self.fc2(features)
            features = self.fc3(features)

            concat = torch.cat((teams, season_emb, features), dim=1)
            out = self.combined_fc(concat)
            return out


season = torch.tensor([2024])
team1 = torch.tensor([1381])
team2 = torch.tensor([1109])

model = MemoryNet(num_teams=1481, num_seasons=2025, input_dim=32)

model.load_state_dict(torch.load('model_weights.pth'))

with torch.no_grad():
    model.eval()
    pred = model(season, team1, team2, infer=True)
