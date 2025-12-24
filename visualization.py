# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# def plot_confusion_matrix(y_true, y_pred, labels, title, save_path=None):
#     cm = confusion_matrix(y_true, y_pred)
#     disp = ConfusionMatrixDisplay(cm, display_labels=labels)

#     disp.plot(cmap="Blues")
#     plt.title(title)
#     plt.tight_layout()

#     if save_path:
#         plt.savefig(save_path, dpi=300)

#     plt.show()

# visualization.py

# import matplotlib.pyplot as plt
# from sklearn.metrics import ConfusionMatrixDisplay

# def plot_confusion_matrix(y_true, y_pred, labels, save_path):
#     disp = ConfusionMatrixDisplay.from_predictions(
#         y_true, y_pred,
#         display_labels=labels,
#         cmap="Blues"
#     )
#     plt.title("SVM Emotion Detection")
#     plt.savefig(save_path)
#     plt.close()

# visualization.py
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)


def plot_confusion_matrix(y_true, y_pred, labels, save_path):
    """
    Plot dan simpan confusion matrix
    """
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=labels,
        cmap="Blues",
        values_format="d"
    )
    plt.title("SVM Emotion Detection")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def print_evaluation_metrics(y_true, y_pred, class_names):
    """
    Hitung dan tampilkan Accuracy, Precision, Recall, F1-score
    """

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print("\n📊 Evaluation Metrics (Validation Set)")
    print("-------------------------------------")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1-score  : {f1:.4f}")

    print("\n📄 Classification Report")
    print("-------------------------------------")
    print(classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    ))

    return acc, prec, rec, f1

