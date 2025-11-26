"""
Create training dataset with historical features for Singapore GP predictions.
Generates driver and team statistics based on past Singapore GP performances.
"""
import pandas as pd

# Load Singapore GP data
df = pd.read_csv("data/merged_singapore.csv")

# Clean position data
df['position'] = pd.to_numeric(df['position'], errors='coerce')
df = df.dropna(subset=['position'])
df['position'] = df['position'].astype(int)
df = df.sort_values(['year', 'driverId']).reset_index(drop=True)

# Create win flag
df['win'] = (df['position'] == 1).astype(int)

# Training data list
training_data = []

# For each race, calculate historical features
for race_year in sorted(df['year'].unique()):
    current_race = df[df['year'] == race_year].copy()
    history = df[df['year'] < race_year].copy()
    
    # Skip first year (no history)
    if len(history) == 0:
        continue
    
    # For each driver in current race
    for _, driver_row in current_race.iterrows():
        driver = driver_row['driver_name']
        team = driver_row['team_name']
        
        # Driver historical stats
        driver_history = history[history['driver_name'] == driver]
        if len(driver_history) > 0:
            last3 = driver_history.tail(3)
            races_count = len(driver_history)
            wins_count = driver_history['win'].sum()
            avg_finish = driver_history['position'].mean()
            best_finish = driver_history['position'].min()
            podiums = (driver_history['position'] <= 3).sum()
            recent_avg = last3['position'].mean()
            recent_podiums = (last3['position'] <= 3).sum()
        else:
            # New driver defaults
            races_count = wins_count = podiums = recent_podiums = 0
            avg_finish = recent_avg = 15.0
            best_finish = 20
        
        # Team historical stats
        team_history = history[history['team_name'] == team]
        if len(team_history) > 0:
            team_races = len(team_history)
            team_wins = team_history['win'].sum()
            team_winrate = team_wins / team_races
            team_avg_finish = team_history['position'].mean()
            team_podiums = (team_history['position'] <= 3).sum()
        else:
            # New team defaults
            team_races = team_wins = team_podiums = 0
            team_winrate = 0.0
            team_avg_finish = 15.0
        
        # Append training sample
        training_data.append({
            'year': race_year,
            'driver_name': driver,
            'team_name': team,
            'driver_sg_races': races_count,
            'driver_sg_wins': wins_count,
            'driver_sg_avg_finish': avg_finish,
            'driver_sg_best_finish': best_finish,
            'driver_sg_podiums': podiums,
            'driver_sg_recent_avg': recent_avg,
            'driver_sg_recent_podiums': recent_podiums,
            'team_sg_races': team_races,
            'team_sg_wins': team_wins,
            'team_sg_winrate': team_winrate,
            'team_sg_avg_finish': team_avg_finish,
            'team_sg_podiums': team_podiums,
            'win': driver_row['win']
        })

# Create DataFrame and save
training_df = pd.DataFrame(training_data)
training_df.to_csv("data/training_dataset.csv", index=False)

print(f"Training dataset created: {len(training_df)} samples, "
      f"{training_df['year'].min()}-{training_df['year'].max()}, "
      f"{training_df['year'].nunique()} races, {training_df['win'].sum()} wins")
