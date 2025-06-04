import os
import json
import pandas as pd
from datetime import datetime
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

df = pd.read_csv("data/bikes.csv")
X = df.iloc[:, :-1]
y_true = df.iloc[:, -1]

model = joblib.load('model.joblib')
y_pred = model.predict(X)

metrics = {
    "timestamp": datetime.utcnow().isoformat(),
    "r2_score": r2_score(y_true, y_pred),
    "mae": mean_absolute_error(y_true, y_pred)
}

with open("metrics_report.json", "w") as f:
    json.dump(metrics, f, indent=4)

log_file = "metrics_log_bikes.csv"
df_log = pd.DataFrame([metrics])
if os.path.exists(log_file):
    df_log.to_csv(log_file, mode='a', header=False, index=False)
else:
    df_log.to_csv(log_file, index=False)

print("Metrics logged to metrics_report.json and metrics_log_bikes.csv")