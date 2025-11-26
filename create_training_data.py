import pandas as pd
import numpy as np

# Load merged Singapore GP data
df = pd.read_csv("data/merged_singapore.csv")
df['position'] = pd.to_numeric(df['position'], errors='coerce')
df = df.dropna(subset=['position'])
df['position'] = df['position'].astype(int)
df = df.sort_values(['year', 'driverId']).reset_index(drop=True)

# Create target: 1 if won (position == 1), 0 otherwise
df['win'] = (df['position'] == 1).astype(int)

# For each race, create features based on PRIOR history only
training_data = []

for race_year in sorted(df['year'].unique()):
    # Get current race drivers
    current_race = df[df['year'] == race_year].copy()
    
    # Get historical data (before this race)
    history = df[df['year'] < race_year].copy()
    
    if len(history) == 0:
        continue  # Skip first year (no history)
    
    for _, driver_row in current_race.iterrows():
        driver = driver_row['driver_name']
        team = driver_row['team_name']
        actual_win = driver_row['win']
        
        # Driver historical stats at Singapore
        driver_history = history[history['driver_name'] == driver]
        
        if len(driver_history) > 0:
            races_count = len(driver_history)
            wins_count = driver_history['win'].sum()
            avg_finish = driver_history['position'].mean()
            best_finish = driver_history['position'].min()
            podiums = (driver_history['position'] <= 3).sum()
            
            # Recent form (last 3 Singapore races)
            last3 = driver_history.tail(3)
            recent_avg = last3['position'].mean()
            recent_podiums = (last3['position'] <= 3).sum()
        else:
            # New driver at Singapore
            races_count = 0
            wins_count = 0
            avg_finish = 15.0  # neutral starting position
            best_finish = 20
            podiums = 0
            recent_avg = 15.0
            recent_podiums = 0
        
        # Team historical stats at Singapore
        team_history = history[history['team_name'] == team]
        if len(team_history) > 0:
            team_races = len(team_history)
            team_wins = team_history['win'].sum()
            team_winrate = team_wins / team_races
            team_avg_finish = team_history['position'].mean()
            team_podiums = (team_history['position'] <= 3).sum()
        else:
            team_races = 0
            team_wins = 0
            team_winrate = 0.0
            team_avg_finish = 15.0
            team_podiums = 0
        
        training_data.append({
            'year': race_year,
            'driver_name': driver,
            'team_name': team,
            # Driver features
            'driver_sg_races': races_count,
            'driver_sg_wins': wins_count,
            'driver_sg_avg_finish': avg_finish,
            'driver_sg_best_finish': best_finish,
            'driver_sg_podiums': podiums,
            'driver_sg_recent_avg': recent_avg,
            'driver_sg_recent_podiums': recent_podiums,
            # Team features
            'team_sg_races': team_races,
            'team_sg_wins': team_wins,
            'team_sg_winrate': team_winrate,
            'team_sg_avg_finish': team_avg_finish,
            'team_sg_podiums': team_podiums,
            # Target
            'win': actual_win
        })

# Create DataFrame
training_df = pd.DataFrame(training_data)

# Save training dataset
training_df.to_csv("data/training_dataset.csv", index=False)
print(f"Training dataset created: {len(training_df)} driver-race combinations")
print(f"Years covered: {training_df['year'].min()} to {training_df['year'].max()}")
print(f"Total races: {training_df['year'].nunique()}")
print(f"Total wins in dataset: {training_df['win'].sum()}")
print(f"\nSample data:")
print(training_df.head(10))
