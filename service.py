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

model = bentoml.sklearn.load_model(
    "health_insurance_anomaly_detector:latest"
)

@bentoml.service(name="health_insurance_anomaly_detection_service")
class InsuranceService:

    @bentoml.api
    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        df = pd.DataFrame(data)
        preds = model.predict(df)
        return {"predictions": preds.tolist()}

