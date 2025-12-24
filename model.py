# from sklearn.svm import SVC

# def build_model():
#     print("[Model] Training SVM classifier")
#     return SVC(
#         kernel="rbf",
#         C=1.0,
#         gamma="scale",
#         probability=True
#     )

# from sklearn.svm import SVC
# import joblib
# import os

# MODEL_PATH = "result/svm_eeg_model.joblib"

# def build_model():
#     print("[Model] Building SVM classifier")
#     return SVC(
#         kernel="rbf",
#         C=1.0,
#         gamma="scale",
#         probability=True
#     )

# def save_model(model):
#     os.makedirs("result", exist_ok=True)
#     joblib.dump(model, MODEL_PATH)
#     print(f"[Model] Model saved to {MODEL_PATH}")

# def load_model():
#     print(f"[Model] Loading model from {MODEL_PATH}")
#     return joblib.load(MODEL_PATH)

# from sklearn.svm import SVC

# def build_model():
#     print("[Model] Training SVM classifier")
#     return SVC(
#         kernel="rbf",
#         C=1.0,
#         gamma="scale",
#         probability=True
#     )
# model.py

from sklearn.svm import SVC

def build_svm():
    return SVC(
        kernel="linear",
        C=1.0,
        probability=True,
        random_state=42
    )
