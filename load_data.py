import pandas as pd

DATA_DIR = "Datasets"

races = pd.read_csv(f"{DATA_DIR}/races.csv")
results = pd.read_csv(f"{DATA_DIR}/results.csv")
drivers = pd.read_csv(f"{DATA_DIR}/drivers.csv")
constructors = pd.read_csv(f"{DATA_DIR}/constructors.csv")
circuits = pd.read_csv(f"{DATA_DIR}/circuits.csv")

print(f"Loaded - Races: {races.shape[0]}, Results: {results.shape[0]}, Drivers: {drivers.shape[0]}")
