import os
import pickle
import click
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Load environment variables - try .env.local first (local dev), then .env (Docker)
load_dotenv(".env.local")  # Local development with localhost
load_dotenv()  # Docker Compose with service names
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from datetime import datetime


def dump_pickle(obj, filename: str):
    """Save object to pickle file."""
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)


def read_energy_data(data_folder: str) -> pd.DataFrame:
    """Read and merge renewable energy data from multiple CSV files.
    
    Loads data from:
    - productie_comnbined.csv: energy production data
    - v_wind_alles_compleet.csv: wind speed data
    - sun_combined.csv: solar radiation data
    
    Returns a DataFrame with columns:
    - tijd: timestamp
    - wind_speed_kmh: wind speed in km/h
    - solar_radiation_wm2: solar radiation in W/m²
    - production_mw: energy production in MW (target)
    """
    # Load production data
    prod_file = os.path.join(data_folder, 'productie_comnbined.csv')
    df_prod = pd.read_csv(prod_file)
    df_prod['tijd'] = pd.to_datetime(df_prod['tijd'], utc=True)
    # Use total production (sum of zon and wind)
    df_prod['production_mw'] = (df_prod['vlaanderen zon kwh'] + df_prod['vlaanderen wind kwh']) / 1000
    df_prod = df_prod[['tijd', 'production_mw']].copy()
    
    # Load wind data
    wind_file = os.path.join(data_folder, 'v_wind_alles_compleet.csv')
    df_wind = pd.read_csv(wind_file)
    df_wind.rename(columns={'tijdstip': 'tijd'}, inplace=True)
    df_wind['tijd'] = pd.to_datetime(df_wind['tijd'], utc=True)
    # Use the first available wind speed column that has data
    df_wind['wind_speed_kmh'] = df_wind['wind_ecmwf_2026'].fillna(df_wind['wind_kmi_2002'])
    df_wind = df_wind[['tijd', 'wind_speed_kmh']].copy()
    
    # Load solar radiation data
    sun_file = os.path.join(data_folder, 'sun_combined.csv')
    df_sun = pd.read_csv(sun_file)
    df_sun.rename(columns={'datum': 'tijd'}, inplace=True)
    df_sun['tijd'] = pd.to_datetime(df_sun['tijd'], utc=True)
    # Use the first available radiation column that has data
    df_sun['solar_radiation_wm2'] = df_sun['open_meteo_radiation'].fillna(df_sun['kmi_radiation_avg'])
    df_sun = df_sun[['tijd', 'solar_radiation_wm2']].copy()
    
    # Merge all dataframes on time using outer merge to preserve all data
    df = df_prod.copy()
    df = df.merge(df_wind, on='tijd', how='outer')
    df = df.merge(df_sun, on='tijd', how='outer')
    
    # Sort by time
    df = df.sort_values('tijd').reset_index(drop=True)
    
    # Fill missing values using forward fill and backward fill
    df['wind_speed_kmh'] = df['wind_speed_kmh'].fillna(method='ffill').fillna(method='bfill')
    df['solar_radiation_wm2'] = df['solar_radiation_wm2'].fillna(method='ffill').fillna(method='bfill')
    df['production_mw'] = df['production_mw'].fillna(method='ffill').fillna(method='bfill')
    
    # Validate required columns
    required_cols = ['wind_speed_kmh', 'solar_radiation_wm2', 'production_mw']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Remove rows with remaining missing values
    df = df.dropna(subset=required_cols)
    
    # Remove outliers (all values should be non-negative)
    df = df[df['production_mw'] >= 0]
    df = df[df['wind_speed_kmh'] >= 0]
    df = df[df['solar_radiation_wm2'] >= 0]
    
    return df


def preprocess_energy_data(df: pd.DataFrame, scaler: StandardScaler = None, fit_scaler: bool = False):
    """Preprocess renewable energy data for model training.
    
    Performs feature engineering and scaling for time-series forecasting.
    
    Args:
        df: DataFrame with energy production and weather features
        scaler: StandardScaler instance for normalization
        fit_scaler: Whether to fit the scaler (True for training data)
    
    Returns:
        X: Scaled feature matrix
        y: Target values (production in MW)
        scaler: Fitted scaler instance
    """
    # Select features and target
    feature_cols = ['wind_speed_kmh', 'solar_radiation_wm2']
    target_col = 'production_mw'
    
    X = df[feature_cols].values
    y = df[target_col].values
    
    # Scale features for better model performance
    if scaler is None:
        scaler = StandardScaler()
    
    if fit_scaler:
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)
    
    return X, y, scaler


@click.command()
@click.option(
    "--data_folder",
    default="../data",
    help="Path to the folder containing CSV files with raw renewable energy data"
)
@click.option(
    "--dest_path",
    default="./output",
    help="Location where the processed data will be saved"
)
@click.option(
    "--train_size",
    default=0.6,
    type=float,
    help="Proportion of data to use for training"
)
@click.option(
    "--val_size",
    default=0.2,
    type=float,
    help="Proportion of data to use for validation (test gets the remainder)"
)
def run_data_prep(data_folder: str, dest_path: str, train_size: float = 0.6, val_size: float = 0.2):
    """Preprocess renewable energy production data for model training.
    
    This script:
    1. Loads raw energy production, weather, and solar radiation data from CSV files
    2. Merges the data on timestamps
    3. Performs feature engineering and validation
    4. Scales features using StandardScaler
    5. Splits data into train, validation, and test sets
    6. Saves preprocessed data and scaler for reproducible model training
    
    The data is time-series aware and maintains temporal order.
    """
    print(f"\n=== Data Preprocessing Pipeline ===")
    print(f"Loading data from: {data_folder}")
    
    # Load and validate data
    df = read_energy_data(data_folder)
    print(f"Loaded {len(df)} records")
    print(f"Date range: {df['tijd'].min()} to {df['tijd'].max()}")
    
    # Preprocess and scale features
    X, y, scaler = preprocess_energy_data(df, fit_scaler=True)
    print(f"\nFeatures shape: {X.shape}")
    print(f"Target statistics (MW):")
    print(f"  Min: {y.min():.2f}, Max: {y.max():.2f}, Mean: {y.mean():.2f}")
    
    # Split data maintaining temporal order (time-series split)
    n_samples = len(X)
    train_idx = int(n_samples * train_size)
    val_idx = int(n_samples * (train_size + val_size))
    
    X_train, y_train = X[:train_idx], y[:train_idx]
    X_val, y_val = X[train_idx:val_idx], y[train_idx:val_idx]
    X_test, y_test = X[val_idx:], y[val_idx:]
    
    print(f"\nData split:")
    print(f"  Train: {len(X_train)} samples ({train_size*100:.1f}%)")
    print(f"  Validation: {len(X_val)} samples ({val_size*100:.1f}%)")
    print(f"  Test: {len(X_test)} samples ({(1-train_size-val_size)*100:.1f}%)")
    
    # Create destination directory
    os.makedirs(dest_path, exist_ok=True)
    
    # Save preprocessed data
    dump_pickle(scaler, os.path.join(dest_path, "scaler.pkl"))
    dump_pickle((X_train, y_train), os.path.join(dest_path, "train.pkl"))
    dump_pickle((X_val, y_val), os.path.join(dest_path, "val.pkl"))
    dump_pickle((X_test, y_test), os.path.join(dest_path, "test.pkl"))
    
    print(f"\nPreprocessed data saved to: {dest_path}")
    print(f"Timestamp: {datetime.now().isoformat()}")


if __name__ == '__main__':
    run_data_prep()
