import pandas as pd

DATA_DIR = "Datasets"

# Load datasets
races = pd.read_csv(f"{DATA_DIR}/races.csv")
results = pd.read_csv(f"{DATA_DIR}/results.csv")
drivers = pd.read_csv(f"{DATA_DIR}/drivers.csv")
constructors = pd.read_csv(f"{DATA_DIR}/constructors.csv")
circuits = pd.read_csv(f"{DATA_DIR}/circuits.csv")

# Display summary
print(f"Loaded datasets - Races: {races.shape}, Results: {results.shape}, Drivers: {drivers.shape}")
print(drivers.head())
