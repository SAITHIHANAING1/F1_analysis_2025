import pandas as pd

# Load and prepare data
df = pd.read_csv("data/merged_singapore.csv")
df['position'] = pd.to_numeric(df['position'], errors='coerce')
df = df.dropna(subset=['position'])
df['position'] = df['position'].astype(int)
df['win'] = (df['position'] == 1).astype(int)

# Historical performance features
races_count = df.groupby('driver_name')['raceId'].count().rename('sg_race_count')
wins_count = df.groupby('driver_name')['win'].sum().rename('sg_win_count')
avg_finish = df.groupby('driver_name')['position'].mean().rename('sg_avg_finish')

# Recent form (last 3 races)
recent_stats = []
for driver, subdf in df.groupby('driver_name'):
    last3 = subdf.sort_values('year').tail(3)
    recent_stats.append({
        'driver_name': driver,
        'sg_last3_avg_finish': last3['position'].mean(),
        'sg_last3_podiums': (last3['position'] <= 3).sum()
    })
recent = pd.DataFrame(recent_stats).set_index('driver_name')

# Team strength
team_agg = df.groupby('team_name').agg({'win': 'sum', 'raceId': 'count'})
team_winrate = (team_agg['win'] / team_agg['raceId']).rename('team_sg_winrate')

# Combine features
features = (pd.DataFrame({'driver_name': df['driver_name'].unique()})
            .merge(races_count, on='driver_name', how='left')
            .merge(wins_count, on='driver_name', how='left')
            .merge(avg_finish, on='driver_name', how='left')
            .merge(recent, on='driver_name', how='left'))

# Add team information
driver_team = df.groupby('driver_name')['team_name'].agg(lambda x: x.value_counts().idxmax()).rename('team_name')
features = (features
            .merge(driver_team, on='driver_name', how='left')
            .merge(team_winrate, on='team_name', how='left'))

features.fillna(features.mean(numeric_only=True), inplace=True)

# Save dataset
features.to_csv("data/feature_dataset.csv", index=False)
print(f"Feature dataset saved: {features.shape[0]} drivers, {features.shape[1]} features")
