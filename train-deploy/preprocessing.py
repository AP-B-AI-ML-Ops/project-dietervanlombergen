import os
import pickle

import numpy as np
import pandas as pd
from prefect import flow, task


@task(name="Load and Format Data")
def load_raw_data(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the raw sun and production datasets and fix their timestamps."""
    sun_df = pd.read_csv(
        os.path.join(data_dir, "sun_combined.csv"), parse_dates=["datum"]
    )
    sun_df["tijd"] = pd.to_datetime(sun_df["datum"], utc=True)

    # Fill missing values directly across your radiation options
    sun_df["radiation_wm2"] = (
        sun_df["open_meteo_radiation"]
        .fillna(sun_df["kmi_radiation_avg"])
        .fillna(sun_df["kaggle_radiation_avg"])
    )
    sun_df = sun_df[["tijd", "radiation_wm2"]]

    prod_df = pd.read_csv(
        os.path.join(data_dir, "productie_comnbined.csv"), parse_dates=["tijd"]
    )
    prod_df["tijd"] = pd.to_datetime(prod_df["tijd"], utc=True)
    prod_df["solar_production_mw"] = prod_df["vlaanderen zon kwh"] / 1000
    prod_df = prod_df[["tijd", "solar_production_mw"]]

    return sun_df, prod_df


@task()
def merge_and_clean(sun_df: pd.DataFrame, prod_df: pd.DataFrame) -> pd.DataFrame:
    """Merge datasets on aligned UTC timestamps and handle missing values."""
    df = pd.merge(prod_df, sun_df, on="tijd", how="inner").sort_values("tijd")
    return df.ffill().bfill().dropna().reset_index(drop=True)


@task()
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclic sine/cosine hour transformations for XGBoost."""
    hours = df["tijd"].dt.hour
    df["sin_hour"] = np.sin(2 * np.pi * hours / 24)
    df["cos_hour"] = np.cos(2 * np.pi * hours / 24)

    # Enforce hard zero rule for night hours when radiation is gone
    df.loc[df["radiation_wm2"] == 0, "solar_production_mw"] = 0.0
    return df


@task()
def split_and_save_pkl(df: pd.DataFrame, output_dir: str, train_ratio: float = 0.8):
    """Split data sequentially and save as features (X) and targets (y) tuples."""
    features = ["radiation_wm2", "sin_hour", "cos_hour"]
    target = "solar_production_mw"

    X = df[features].values
    y = df[target].values

    # Chronological time-series split
    split_idx = int(len(df) * train_ratio)
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_val, y_val = X[split_idx:], y[split_idx:]

    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "train.pkl"), "wb") as f:
        pickle.dump((X_train, y_train), f)

    with open(os.path.join(output_dir, "val.pkl"), "wb") as f:
        pickle.dump((X_val, y_val), f)

    print(f"✓ Split data successfully saved to {output_dir}")
    print(f"  Train samples: {len(X_train)} | Val samples: {len(X_val)}")


@flow()
def solar_preprocessing_flow(data_dir: str, output_dir: str):

    sun_df, prod_df = load_raw_data(data_dir)
    cleaned_df = merge_and_clean(sun_df, prod_df)
    engineered_df = engineer_features(cleaned_df)
    split_and_save_pkl(engineered_df, output_dir)


if __name__ == "__main__":
    solar_preprocessing_flow(data_dir="data/", output_dir="./output")
