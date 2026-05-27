import mlflow
import pandas as pd
from flask import Flask, jsonify, request

app = Flask("solar-energy-prediction")

mlflow.set_tracking_uri("http://experiment-tracking:5000")
MODEL_NAME = "solar-production-model"
MODEL_VERSION = "1"

print(f"Loading model '{MODEL_NAME}' version {MODEL_VERSION} from registry...")
model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{MODEL_VERSION}")


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    """
    REST API Endpoint accepting weather forecast data inputs
    and returning the predicted solar energy production in MW.
    """
    # 2. Extract JSON payload containing weather forecast details
    forecast_data = request.get_json()

    # 3. Restructure payload into a DataFrame matching training features
    # Expects columns: 'radiation_wm2', 'sin_hour', 'cos_hour'
    input_features = pd.DataFrame(
        [
            {
                "radiation_wm2": float(forecast_data["radiation_wm2"]),
                "sin_hour": float(forecast_data["sin_hour"]),
                "cos_hour": float(forecast_data["cos_hour"]),
            }
        ]
    )

    # 4. Generate prediction using the loaded MLflow model pipeline
    predictions = model.predict(input_features)
    predicted_production_mw = float(predictions[0])

    # 5. Build and return response
    result = {"predicted_energy_production_mw": predicted_production_mw}

    return jsonify(result)


if __name__ == "__main__":
    # Access this inside your Dev Container or local network at port 9696
    app.run(debug=True, host="0.0.0.0", port=9696)
