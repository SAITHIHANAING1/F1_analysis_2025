import pandas as pd

df = pd.read_csv("data/merged_singapore.csv")
df['position'] = pd.to_numeric(df['position'], errors='coerce').dropna().astype(int)
df = df.dropna(subset=['position']).sort_values(['year', 'driverId']).reset_index(drop=True)
df['win'] = (df['position'] == 1).astype(int)

training_data = []

for race_year in sorted(df['year'].unique()):
    current_race = df[df['year'] == race_year].copy()
    history = df[df['year'] < race_year].copy()
    
    if len(history) == 0:
        continue
    
    for _, driver_row in current_race.iterrows():
        driver, team = driver_row['driver_name'], driver_row['team_name']
        driver_history = history[history['driver_name'] == driver]
        
        if len(driver_history) > 0:
            last3 = driver_history.tail(3)
            races_count, wins_count = len(driver_history), driver_history['win'].sum()
            avg_finish, best_finish = driver_history['position'].mean(), driver_history['position'].min()
            podiums = (driver_history['position'] <= 3).sum()
            recent_avg, recent_podiums = last3['position'].mean(), (last3['position'] <= 3).sum()
        else:
            races_count = wins_count = podiums = recent_podiums = 0
            avg_finish = recent_avg = 15.0
            best_finish = 20
        
        team_history = history[history['team_name'] == team]
        if len(team_history) > 0:
            team_races, team_wins = len(team_history), team_history['win'].sum()
            team_winrate = team_wins / team_races
            team_avg_finish = team_history['position'].mean()
            team_podiums = (team_history['position'] <= 3).sum()
        else:
            team_races = team_wins = team_podiums = 0
            team_winrate = 0.0
            team_avg_finish = 15.0
        
        training_data.append({
            'year': race_year, 'driver_name': driver, 'team_name': team,
            'driver_sg_races': races_count, 'driver_sg_wins': wins_count,
            'driver_sg_avg_finish': avg_finish, 'driver_sg_best_finish': best_finish,
            'driver_sg_podiums': podiums, 'driver_sg_recent_avg': recent_avg,
            'driver_sg_recent_podiums': recent_podiums, 'team_sg_races': team_races,
            'team_sg_wins': team_wins, 'team_sg_winrate': team_winrate,
            'team_sg_avg_finish': team_avg_finish, 'team_sg_podiums': team_podiums,
            'win': driver_row['win']
        })

training_df = pd.DataFrame(training_data)
training_df.to_csv("data/training_dataset.csv", index=False)

print(f"Training dataset: {len(training_df)} samples, {training_df['year'].min()}-{training_df['year'].max()}, "
      f"{training_df['year'].nunique()} races, {training_df['win'].sum()} wins")
