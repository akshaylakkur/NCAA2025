import numpy as np
import pandas as pd
import os

'''
Part 1 -> Sorting
1. Seperate by team and all available data per team
2. (Note) All should be the same length hopefully and possibly
'''
male_files = []
female_files = []
universal = []
orig_path = 'march-machine-learning-mania-2025/'

for dir in os.listdir('march-machine-learning-mania-2025'):
    if dir.startswith('M') and dir != 'MTeams.csv':
        male_files.append(dir)
    elif dir.startswith('W') and dir != 'WTeams.csv':
        female_files.append(dir)
    else:
        if dir != 'MTeams.csv' and dir != 'WTeams.csv':
            universal.append(dir)

headers = ['Season', 'DayNum', 'team1', 'team1_score', 'team1_seed', 'team2', 'team2_score', 'team2_seed', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WSH', 'WBlk', 'WPF', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LSH', 'LBlk', 'LPF', 'NumOT']
male_frame, female_frame = pd.DataFrame(columns=headers), pd.DataFrame(columns=headers)
dr = pd.read_csv(orig_path + 'MNCAATourneyDetailedResults.csv')
seeds = pd.read_csv(orig_path + 'MNCAATourneySeeds.csv')

# Merge seeds for the winning team (WTeamID)
dr = dr.merge(
    seeds, how='left', left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID']
).rename(columns={"Seed": "WTeamSeed"})
dr = dr.drop(columns=['TeamID'])

# Merge seeds for the losing team (LTeamID)
dr = dr.merge(
    seeds, how='left', left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID']
).rename(columns={"Seed": "LTeamSeed"})
dr = dr.drop(columns=['TeamID'])


# Reorder columns to place WTeamSeed in position 4 and LTeamSeed in position 7
columns_order = list(dr.columns)  # Get current column order
columns_order.insert(4, columns_order.pop(columns_order.index('WTeamSeed')))  # Move WTeamSeed to position 4
columns_order.insert(7, columns_order.pop(columns_order.index('LTeamSeed')))  # Move LTeamSeed to position 7

# Apply the new column order
dr = dr[columns_order]

dr.to_csv('male_data.csv', index=False)


