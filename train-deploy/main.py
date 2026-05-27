from hpo import solar_hpo_flow
from prefect import flow

# Import the core flow entrypoints from your individual pipeline modules
from preprocessing import solar_preprocessing_flow
from register_model import solar_registration_flow
from train import solar_training_flow


@flow()
def main_training_flow(
    raw_data_dir: str = "data/", artifacts_dir: str = "./output", hpo_trials: int = 15
):

    solar_preprocessing_flow(data_dir=raw_data_dir, output_dir=artifacts_dir)
    solar_training_flow(data_dir=artifacts_dir)
    solar_hpo_flow(data_dir=artifacts_dir, n_trials=hpo_trials)
    solar_registration_flow(data_dir=artifacts_dir)


if __name__ == "__main__":
    main_training_flow(raw_data_dir="data/", artifacts_dir="./output", hpo_trials=15)
