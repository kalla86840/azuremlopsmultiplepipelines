import json
import os
import pandas as pd
import joblib
from sklearn.metrics import r2_score, mean_absolute_error

df = pd.read_csv("data/house.csv")
X = df.iloc[:, :-1]
y_true = df.iloc[:, -1]

model = joblib.load('model.joblib')
y_pred = model.predict(X)

metrics = {
    "r2_score": r2_score(y_true, y_pred),
    "mae": mean_absolute_error(y_true, y_pred)
}

with open("metrics_report.json", "r") as f:
    print("Evaluation metrics (latest run):")
    print(json.dumps(metrics, indent=4))

baseline_path = "baseline_metrics.json"
if os.path.exists(baseline_path):
    with open(baseline_path) as f:
        baseline = json.load(f)
    if metrics["mae"] < baseline["mae"]:
        print("New model is better. Updating baseline.")
        with open(baseline_path, "w") as f:
            json.dump(metrics, f, indent=4)
    else:
        print("New model did not outperform the baseline.")
else:
    print("No baseline found. Creating one.")
    with open(baseline_path, "w") as f:
        json.dump(metrics, f, indent=4)