import os
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
import mlflow
from prefect import task, flow


@task()
def load_final_data(output_dir: str) -> tuple:

    with open(os.path.join(output_dir, "train.pkl"), "rb") as f:
        X_train, y_train = pickle.load(f)
        
    with open(os.path.join(output_dir, "val.pkl"), "rb") as f:
        X_val, y_val = pickle.load(f)
        
    return X_train, y_train, X_val, y_val


@task()
def train_and_register(X_train, y_train, X_val, y_val):

    mlflow.set_tracking_uri("http://experiment-tracking:5000") 
    mlflow.set_experiment("solar-production")

    X_final = np.concatenate([X_train, X_val], axis=0)
    y_final = np.concatenate([y_train, y_val], axis=0)

    features = ["radiation_wm2", "sin_hour", "cos_hour"]
    X_final_df = pd.DataFrame(X_final, columns=features)
    y_final_series = pd.Series(y_final, name="solar_production_mw")

    best_params = {
        "n_estimators": 120,    
        "max_depth": 5,         
        "learning_rate": 0.03,  
        "random_state": 42
    }

    mlflow.xgboost.autolog(log_models=True)

    with mlflow.start_run() as run:
        model = xgb.XGBRegressor(**best_params)
        model.fit(X_final_df, y_final_series, verbose=False)
        
        model_uri = f"runs:/{run.info.run_id}/model"
        
        model_details = mlflow.register_model(
            model_uri=model_uri,
            name="solar-production-model"
        )
        
        print(f"\n✓ Model successfully registered!")
        print(f"✓ Registry Name: {model_details.name}")
        print(f"✓ Model Version: Version {model_details.version}")


@flow()
def solar_registration_flow(data_dir: str = "./output"):

    X_train, y_train, X_val, y_val = load_final_data(data_dir)
    train_and_register(X_train, y_train, X_val, y_val)


if __name__ == "__main__":

    solar_registration_flow(data_dir="./output")