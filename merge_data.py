import pandas as pd

DATA_DIR, OUT_DIR = "Datasets", "data"

races = pd.read_csv(f"{DATA_DIR}/races.csv")
results = pd.read_csv(f"{DATA_DIR}/results.csv")
drivers = pd.read_csv(f"{DATA_DIR}/drivers.csv")
constructors = pd.read_csv(f"{DATA_DIR}/constructors.csv").rename(columns={'name': 'team_name'})
circuits = pd.read_csv(f"{DATA_DIR}/circuits.csv")

df = (results
      .merge(races[['raceId', 'year', 'name', 'circuitId']], on='raceId')
      .merge(drivers[['driverId', 'forename', 'surname']], on='driverId')
      .merge(constructors[['constructorId', 'team_name']], on='constructorId')
      .merge(circuits[['circuitId', 'circuitRef', 'country']], on='circuitId'))

df['driver_name'] = df['forename'] + ' ' + df['surname']
singapore_df = df[df['name'].str.contains("Singapore", case=False, na=False)].copy()
singapore_df = singapore_df.drop(columns=['number', 'positionText', 'url', 'forename', 
                                           'surname', 'circuitRef'], errors='ignore')

singapore_df.to_csv(f"{OUT_DIR}/merged_singapore.csv", index=False)
print(f"Singapore dataset: {singapore_df.shape[0]} rows, {singapore_df.shape[1]} columns")
