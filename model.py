import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

MODEL_DIR = "result/models"

def build_svm_linear():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="linear",
            C=1.0,
            probability=True,
            class_weight="balanced",
            random_state=42
        ))
    ])

def train_and_save(model, X, y, name):
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.fit(X, y)
    joblib.dump(model, f"{MODEL_DIR}/{name}.pkl")
    print(f"Model saved: {MODEL_DIR}/{name}.pkl")

def load_model(name):
    return joblib.load(f"{MODEL_DIR}/{name}.pkl")
