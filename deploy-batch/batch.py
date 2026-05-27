import os
import pickle

import mlflow
import pandas as pd
from mlflow import MlflowClient
from prefect import flow, task
from prefect.client.schemas.schedules import CronSchedule

# 1. Setup global infrastructure addresses
MLFLOW_TRACKING_URI = "http://experiment-tracking:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient(MLFLOW_TRACKING_URI)
MODEL_NAME = "solar-production-model"


@task()
def read_dataframe(filename: str) -> tuple:
    """Read binary preprocessed validation pickle data arrays."""
    with open(filename, "rb") as f:
        X_val, y_val = pickle.load(f)
    return X_val, y_val


@task()
def get_latest_version(model_name: str) -> str:
    """Use native MlflowClient to extract the highest active version index number."""
    latest_versions = client.get_latest_versions(name=model_name)
    if not latest_versions:
        raise ValueError(
            f"No registered versions found for model matching name: '{model_name}'"
        )
    return latest_versions[-1].version


@task()
def load_production_model(latest_version: str):
    """Retrieve the targeted serialized pipeline wrapper from the artifacts hub."""
    print(f"...loading model {MODEL_NAME} version {latest_version}")
    model_uri = f"models:/{MODEL_NAME}/{latest_version}"
    return mlflow.pyfunc.load_model(model_uri)


@flow()
def run_batch(
    input_file_path: str = "./output/val.csv",
    output_base_dir: str = "/batch-data/report",
):
    """
    Prefect Flow evaluating scheduled batch records against the registered champion model,
    saving delta verification frames down to structural disk storage paths.
    """
    # 1. Get registry version and load model asset
    latest_version = get_latest_version(MODEL_NAME)
    model = load_production_model(latest_version)

    # 2. Extract batch dataset target inputs from the unpacked pickle arrays
    print("...reading database validation arrays")
    X_batch, y_actual = read_dataframe(input_file_path)

    # 3. FIXED: Convert array to DataFrame to satisfy the MLflow model schema signature
    features = ["radiation_wm2", "sin_hour", "cos_hour"]
    X_batch_df = pd.DataFrame(X_batch, columns=features)

    # 4. Generate batch inferences using the mapped DataFrame
    print("...calculating energy production estimations")
    y_pred = model.predict(X_batch_df)

    # 5. Compile consolidated structural validation report dataframe
    print("...building reporting delta matrices")
    df_result = pd.DataFrame()
    df_result["solar_production_mw_actual"] = y_actual
    df_result["solar_production_mw_predicted"] = y_pred
    df_result["prediction_error_delta"] = (
        df_result["solar_production_mw_actual"]
        - df_result["solar_production_mw_predicted"]
    )
    df_result["model_run_id"] = model.metadata.run_id
    df_result["model_version"] = latest_version

    # 6. Save report cleanly back out as Parquet
    print("...serializing batch tracking report outputs")

    from datetime import datetime

    now = datetime.now()
    year, month = now.year, now.month

    path = os.path.join(output_base_dir, "solar", f"{year:04d}", f"{month:02d}")
    os.makedirs(path, exist_ok=True)

    output_filename = f"{path}/{model.metadata.run_id}.parquet"
    df_result.to_parquet(output_filename, index=False)
    print(f"✓ Batch file successfully written down to path: {output_filename}")


if __name__ == "__main__":
    run_batch.serve(
        name="solar-monthly-batch-deployment",
        schedule=CronSchedule(cron="0 0 * * *", timezone="Europe/Brussels"),
        parameters={
            "input_file_path": "train-deploy/output/val.pkl",
            "output_base_dir": "./batch-reports",
        },
    )
