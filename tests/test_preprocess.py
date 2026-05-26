# tests/test_preprocess.py
import os
import pickle

import pandas as pd
import pytest
from preprocessing import (
    engineer_features,
    load_raw_data,
    merge_and_clean,
    split_and_save_pkl,
)

# Define paths to your real workspace inputs
DATA_DIR = "./data/"
SUN_FILE = os.path.join(DATA_DIR, "sun_combined.csv")
PROD_FILE = os.path.join(DATA_DIR, "productie_comnbined.csv")


@pytest.fixture(scope="module")
def ensure_raw_assets_exist():
    """Safety wrapper ensuring file paths exist before starting the test loop."""
    if not os.path.exists(SUN_FILE) or not os.path.exists(PROD_FILE):
        pytest.fail(
            f"Missing required production source data files in '{DATA_DIR}'. "
            "Please ensure 'sun_combined.csv' and 'productie_comnbined.csv' are present."
        )


def test_load_raw_data_parses_production_files(ensure_raw_assets_exist):
    """Test that source files load correctly and execute base scaling calculations."""
    sun_df, prod_df = load_raw_data(DATA_DIR)

    # Assert columns were renamed and engineered correctly by the loader
    assert "radiation_wm2" in sun_df.columns
    assert "solar_production_mw" in prod_df.columns

    # Assert data types are correct
    assert pd.api.types.is_datetime64_any_dtype(sun_df["tijd"])
    assert pd.api.types.is_datetime64_any_dtype(prod_df["tijd"])

    # Validate that conversion scaling (kWh -> MW) worked and values are numeric
    assert prod_df["solar_production_mw"].notna().any()


def test_merge_and_clean_production_alignment(ensure_raw_assets_exist):
    """Test that inner join successfully aligns weather and production timelines."""
    sun_df, prod_df = load_raw_data(DATA_DIR)
    merged_df = merge_and_clean(sun_df, prod_df)

    # Assert data exists post-merge
    assert len(merged_df) > 0

    # Assert no missing target observations remain
    assert merged_df["radiation_wm2"].notna().all()
    assert merged_df["solar_production_mw"].notna().all()


def test_engineer_features_production_bounds(ensure_raw_assets_exist):
    """Confirm cyclical domain features map between -1.0 and 1.0."""
    sun_df, prod_df = load_raw_data(DATA_DIR)
    merged_df = merge_and_clean(sun_df, prod_df)

    fe_df = engineer_features(merged_df)

    # Assert structural engineering layers were added
    assert "sin_hour" in fe_df.columns
    assert "cos_hour" in fe_df.columns

    # Cyclical trigonometric bounds must remain between -1 and 1
    assert fe_df["sin_hour"].min() >= -1.0
    assert fe_df["sin_hour"].max() <= 1.0
    assert fe_df["cos_hour"].min() >= -1.0
    assert fe_df["cos_hour"].max() <= 1.0


def test_split_and_save_pkl_production_dimensions(tmp_path, ensure_raw_assets_exist):
    """Verify final matrix width splits precisely match the 3-feature shape requirement."""
    sun_df, prod_df = load_raw_data(DATA_DIR)
    merged_df = merge_and_clean(sun_df, prod_df)
    fe_df = engineer_features(merged_df)

    # Save the pickle binaries to a temporary directory to avoid overwriting production output
    split_and_save_pkl(fe_df, str(tmp_path), train_ratio=0.8)

    # Load and unpack the generated pickles
    with open(tmp_path / "train.pkl", "rb") as f:
        X_train, y_train = pickle.load(f)
    with open(tmp_path / "val.pkl", "rb") as f:
        X_val, y_val = pickle.load(f)

    # Assert splits actually contain records
    assert len(X_train) > 0
    assert len(X_val) > 0
    assert len(y_train) == len(X_train)
    assert len(y_val) == len(X_val)

    # CRITICAL INFERENCE SIGNATURE VERIFICATION
    # Ensures both datasets have exactly 3 columns: [radiation_wm2, sin_hour, cos_hour]
    assert X_train.shape[1] == 3
    assert X_val.shape[1] == 3
