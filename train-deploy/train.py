import os
import pickle

import mlflow
import xgboost as xgb
from prefect import flow, task


@task()
def load_data(output_dir: str) -> tuple:
    with open(os.path.join(output_dir, "train.pkl"), "rb") as f:
        X_train, y_train = pickle.load(f)

    with open(os.path.join(output_dir, "val.pkl"), "rb") as f:
        X_val, y_val = pickle.load(f)

    return X_train, y_train, X_val, y_val


@task()
def train_and_log(X_train, y_train, X_val, y_val):

    mlflow.set_tracking_uri("http://experiment-tracking:5000")
    mlflow.set_experiment("solar-production")

    # Enable autologging before starting the run
    mlflow.xgboost.autolog(log_models=True)

    with mlflow.start_run(run_name="Prefect_XGBoost_Baseline"):
        model = xgb.XGBRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
        )

        print("Training XGBoost model...")
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        print("✓ Training complete. Metrics and model logged to MLflow!")


@flow()
def solar_training_flow(data_dir: str = "./output"):

    X_train, y_train, X_val, y_val = load_data(data_dir)
    train_and_log(X_train, y_train, X_val, y_val)


if __name__ == "__main__":
    solar_training_flow(data_dir="./output")
