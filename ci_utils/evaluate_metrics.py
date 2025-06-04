import os
import json
import pandas as pd
from datetime import datetime
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

# Read dataset
df = pd.read_csv('data/cars.csv')
X = df.iloc[:, :-1]
y_true = df.iloc[:, -1]

# Load model
model = joblib.load('model.joblib')
y_pred = model.predict(X)

# Calculate metrics
metrics = {
    "timestamp": datetime.utcnow().isoformat(),
    "r2_score": r2_score(y_true, y_pred),
    "mae": mean_absolute_error(y_true, y_pred)
}

# Save to JSON (latest run)
with open("metrics_report.json", "w") as f:
    json.dump(metrics, f, indent=4)

# Append to CSV log (cumulative history)
csv_path = "metrics_log.csv"
df_log = pd.DataFrame([metrics])
if os.path.exists(csv_path):
    df_log.to_csv(csv_path, mode='a', header=False, index=False)
else:
    df_log.to_csv(csv_path, index=False)

print("Metrics recorded to metrics_report.json and metrics_log.csv")