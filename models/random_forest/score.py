import joblib
import json
from azureml.core.model import Model

def init():
    global model
    model_path = Model.get_model_path("random_forest_model")
    model = joblib.load(model_path)

def run(raw_data):
    data = json.loads(raw_data)["data"]
    return model.predict(data).tolist()