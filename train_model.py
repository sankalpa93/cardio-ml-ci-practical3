import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

DATA_FILE = "cardio_train.csv"
MODEL_FILE = "cardio_model.pkl"
METRICS_FILE = "metrics.json"

FEATURES = [
    "age_years", "gender", "height", "weight", "ap_hi", "ap_lo",
    "cholesterol", "gluc", "smoke", "alco", "active",
]

print("Loading dataset...")
df = pd.read_csv(DATA_FILE, sep=";")
print("Raw records:", len(df))

# Convert age from days to years
df["age_years"] = (df["age"] / 365.25).round().astype(int)

# Remove medically impossible blood-pressure values
df = df[(df["ap_hi"] >= 70) & (df["ap_hi"] <= 250)]
df = df[(df["ap_lo"] >= 40) & (df["ap_lo"] <= 200)]
df = df[df["ap_hi"] > df["ap_lo"]]
print("Records after cleaning:", len(df))
df.to_csv("cardio_cleaned.csv", index=False)

X = df[FEATURES]
y = df["cardio"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
)
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))
print("Accuracy:", round(accuracy, 4))

joblib.dump(model, MODEL_FILE)

metrics = {
    "accuracy": float(accuracy),
    "training_records": len(X_train),
    "testing_records": len(X_test),
}
with open(METRICS_FILE, "w") as file:
    json.dump(metrics, file, indent=4)

print("Model and metrics saved.")
