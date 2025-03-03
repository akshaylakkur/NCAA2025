import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import torch.nn.functional as F
import numpy as np
import os

'win team - index = 2'
'losing team - index = 5'

'All Teams'
teams = 'files/MTeams.csv'
mens_teams = pd.read_csv(teams)['TeamID']

'Full Data'
full_data = 'files/full_data.csv'
data = pd.read_csv(full_data)

'Data dict'
def sort_team_data(teams, data):
    team_games = {}

    for team_id in teams:
        # Get all rows where the team appears in either column
        team_matches = data[(data['LTeamID'] == team_id) | (data['WTeamID'] == team_id)].sample(frac=1, random_state=None).reset_index(drop=True)
        if not team_matches.empty:
            team_games[team_id] = team_matches
            team_games[team_id]['Score'] = (team_games[team_id]['WTeamID'] == team_id).astype(int)

            team_games[team_id].to_csv(f'/Users/akshaylakkur/PycharmProjects/MarchMadness/current/indiv_team_data/historic_data_{team_id}.csv', index=False)
    return team_games

team_games = sort_team_data(mens_teams, data)
print(team_games)

