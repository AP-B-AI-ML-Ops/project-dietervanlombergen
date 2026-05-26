import os
import pickle
import optuna
import xgboost as xgb
import mlflow
from sklearn.metrics import root_mean_squared_error
from prefect import task, flow


@task()
def load_data_for_hpo(output_dir: str) -> tuple:

    with open(os.path.join(output_dir, "train.pkl"), "rb") as f:
        X_train, y_train = pickle.load(f)
        
    with open(os.path.join(output_dir, "val.pkl"), "rb") as f:
        X_val, y_val = pickle.load(f)
        
    return X_train, y_train, X_val, y_val


def create_objective(X_train, y_train, X_val, y_val):

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 250),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "random_state": 42
        }
        
        with mlflow.start_run(run_name=f"Trial_{trial.number}", nested=True):
            mlflow.xgboost.autolog(log_models=False, silent=True)
            
            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            
            preds = model.predict(X_val)
            rmse = root_mean_squared_error(y_val, preds)
            
            return rmse
            
    return objective


@task()
def run_hpo_study(X_train, y_train, X_val, y_val, n_trials: int = 15):

    mlflow.set_tracking_uri("http://experiment-tracking:5000") 
    mlflow.set_experiment("solar-production-hpo")

    with mlflow.start_run(run_name="Optuna_XGBoost_Search"):
        objective = create_objective(X_train, y_train, X_val, y_val)
        
        study = optuna.create_study(direction="minimize")
        print(f"Optimizing hyperparameters over {n_trials} trials...")
        study.optimize(objective, n_trials=n_trials)
        
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_rmse", study.best_value)
        
        print("\n✓ Optimization Complete!")
        print(f"Best RMSE: {study.best_value}")
        print(f"Best Params: {study.best_params}")


@flow()
def solar_hpo_flow(data_dir: str = "./output", n_trials: int = 15):

    X_train, y_train, X_val, y_val = load_data_for_hpo(data_dir)
    run_hpo_study(X_train, y_train, X_val, y_val, n_trials=n_trials)


if __name__ == "__main__":

    solar_hpo_flow(data_dir="./output", n_trials=15)