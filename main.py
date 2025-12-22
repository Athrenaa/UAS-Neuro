import os
import sys
sys.path.append(os.path.dirname(__file__))

from load_data import load_data
from preprocessing import preprocess_eeg
from feature_extractin import extract_features
from model import build_svm_linear, train_and_save, load_model
from visualization import plot_confusion_matrix

from sklearn.metrics import accuracy_score, classification_report

# =====================
# LOAD DATA
# =====================
X_train, y_train, X_val, y_val = load_data()

# =====================
# PREPROCESS
# =====================
X_train = preprocess_eeg(X_train)
X_val   = preprocess_eeg(X_val)

# =====================
# FEATURE EXTRACTION
# =====================
X_train_feat = extract_features(X_train)
X_val_feat   = extract_features(X_val)

# =====================
# TRAIN + SAVE
# =====================
svm = build_svm_linear()
train_and_save(svm, X_train_feat, y_train, "svm_linear")

# =====================
# LOAD MODEL
# =====================
model = load_model("svm_linear")

# =====================
# VALIDATION
# =====================
y_pred = model.predict(X_val_feat)

print("Accuracy:", accuracy_score(y_val, y_pred))
print(classification_report(y_val, y_pred))

# =====================
# VISUALIZATION
# =====================
os.makedirs("result/figures", exist_ok=True)

plot_confusion_matrix(
    y_val,
    y_pred,
    labels=["Negative", "Neutral", "Positive"],
    title="Confusion Matrix - SVM Linear",
    save_path="result/figures/svm_confusion.png"
)
