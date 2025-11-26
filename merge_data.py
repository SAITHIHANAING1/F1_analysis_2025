import pandas as pd

DATA_DIR = "Datasets"
OUT_DIR = "data"

# Load raw tables
races = pd.read_csv(f"{DATA_DIR}/races.csv")
results = pd.read_csv(f"{DATA_DIR}/results.csv")
drivers = pd.read_csv(f"{DATA_DIR}/drivers.csv")
constructors = pd.read_csv(f"{DATA_DIR}/constructors.csv").rename(columns={'name': 'team_name'})
circuits = pd.read_csv(f"{DATA_DIR}/circuits.csv")

# Merge all data
df = (results
      .merge(races[['raceId', 'year', 'name', 'circuitId']], on='raceId', how='left')
      .merge(drivers[['driverId', 'forename', 'surname']], on='driverId', how='left')
      .merge(constructors[['constructorId', 'team_name']], on='constructorId', how='left')
      .merge(circuits[['circuitId', 'circuitRef', 'country']], on='circuitId', how='left'))

df['driver_name'] = df['forename'] + ' ' + df['surname']

# Filter for Singapore GP
singapore_df = df[df['name'].str.contains("Singapore", case=False, na=False)].copy()

# Drop unnecessary columns
cols_to_drop = ['number', 'positionText', 'url', 'forename', 'surname', 'circuitRef']
singapore_df = singapore_df.drop(columns=cols_to_drop, errors='ignore')

# Save dataset
singapore_df.to_csv(f"{OUT_DIR}/merged_singapore.csv", index=False)
print(f"Merged Singapore dataset saved ({singapore_df.shape[0]} rows, {singapore_df.shape[1]} columns)")
