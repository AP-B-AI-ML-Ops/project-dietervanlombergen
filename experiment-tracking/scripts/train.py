import os
import pickle
import click
import mlflow
from dotenv import load_dotenv

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

# Load environment variables - try .env.local first (local dev), then .env (Docker)
load_dotenv()  # Docker Compose with service names


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed weather data was saved"
)
def run_train(data_path: str):
    # Get database credentials from environment variables
    postgres_user = os.getenv("POSTGRES_USER", "postgres")
    postgres_password = os.getenv("POSTGRES_PASSWORD", "postgres")
    postgres_host = os.getenv("POSTGRES_HOST", "localhost")
    postgres_host = "127.0.0.1"
    postgres_port = os.getenv("POSTGRES_PORT", "5432")
    postgres_db = os.getenv("POSTGRES_DB", "mlflow_db")
    
    # Set the tracking URI for MLflow using PostgreSQL
    tracking_uri = f"postgresql+psycopg2://{postgres_user}:{postgres_password}@{postgres_host}:{postgres_port}/{postgres_db}"
    print(f"Using MLflow tracking URI: {tracking_uri}")
    mlflow.set_tracking_uri(tracking_uri)
      
    
    # set the experiment for mlflow
    mlflow.set_experiment("renewable-energy-experiment")
    # start an mlflow run
    with mlflow.start_run():
        # set some mlflow tags (e.g. developer)
        mlflow.set_tag("developer", "energy-forecasting")
        # log params in mlflow (e.g. path to the data for training, validation, and test data)
        mlflow.log_param("train-data-path", data_path)
        mlflow.log_param("valid-data-path", data_path)
        mlflow.log_param("test-data-path", data_path)

        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))
        X_test, y_test = load_pickle(os.path.join(data_path, "test.pkl"))

        # move the values for the regressor (below) to a variable
        max_depth = 10
        random_state = 42
        # log the values for the regressor in mlflow
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)

        rf = RandomForestRegressor(max_depth=max_depth, random_state=random_state)
        rf.fit(X_train, y_train)
        
        # Make predictions on validation and test sets
        y_pred_val = rf.predict(X_val)
        y_pred_test = rf.predict(X_test)

        # Calculate RMSE on validation and test sets
        rmse_val = root_mean_squared_error(y_val, y_pred_val)
        rmse_test = root_mean_squared_error(y_test, y_pred_test)

        # log the metrics in mlflow
        mlflow.log_metric("rmse_val", rmse_val)
        mlflow.log_metric("rmse_test", rmse_test)
        
        print(f"Validation RMSE: {rmse_val:.4f}")
        print(f"Test RMSE: {rmse_test:.4f}")

if __name__ == '__main__':
    run_train()
