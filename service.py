# import bentoml
# from bentoml.io import JSON, PandasDataFrame


# # Load the registered model from BentoML registry using the latest version.
# model_runner = bentoml.sklearn.get("health_insurance_anomaly_detector:latest").to_runner()


# # Create a BentoML service and attach the model runner.
# svc = bentoml.Service("health_insurance_anomaly_detection_service", runners=[model_runner])


# # API endpoint for prediction using a Pandas DataFrame as input and JSON as output.
# @svc.api(input=PandasDataFrame(), output=JSON())
# def predict(data):
#     predictions = model_runner.predict.run(data)
#     return {"predictions": predictions.tolist()}

import bentoml
import pandas as pd
from typing import Dict, Any


# Load model from BentoML model store
model = bentoml.sklearn.get(
    "health_insurance_anomaly_detector:latest"
).load_model()


@bentoml.service(
    name="health_insurance_anomaly_detection_service"
)
class HealthInsuranceService:

    @bentoml.api
    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Expects JSON input that can be converted to a DataFrame
        Example:
        {
            "age": [45],
            "bmi": [28.5],
            "children": [2],
            "charges": [12000]
        }
        """
        df = pd.DataFrame(data)
        predictions = model.predict(df)
        return {"predictions": predictions.tolist()}
