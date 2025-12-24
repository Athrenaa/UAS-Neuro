# import os
# import scipy.io as sio
# import numpy as np

# LABEL_MAP = {
#     "Negative": 0,
#     "Neutral": 1,
#     "Positive": 2
# }

# def extract_eeg_from_mat(mat):
#     for key in mat.keys():
#         if key.startswith("__"):
#             continue
#         data = mat[key]
#         if isinstance(data, np.ndarray) and data.ndim in [2, 3]:
#             return data
#     raise ValueError("No EEG data found")

# def load_split(base_path):
#     X, y = [], []
#     lengths = []

#     for label_name, label_id in LABEL_MAP.items():
#         folder = os.path.join(base_path, label_name)

#         for file in os.listdir(folder):
#             if file.endswith(".mat"):
#                 mat = sio.loadmat(os.path.join(folder, file))
#                 eeg = extract_eeg_from_mat(mat)

#                 if eeg.ndim == 3:
#                     eeg = eeg.squeeze()

#                 X.append(eeg)
#                 y.append(label_id)
#                 lengths.append(eeg.shape[1])

#                 print(f"Loaded {file} | shape={eeg.shape}")

#     # 🔑 CROP KE PANJANG MINIMUM
#     min_len = min(lengths)
#     print(f"Cropping all signals to length: {min_len}")

#     X_cropped = [eeg[:, :min_len] for eeg in X]

#     return np.array(X_cropped), np.array(y)

# def load_data():
#     print("[1] Loading TRAIN data")
#     X_train, y_train = load_split("dataset/raw")

#     print("\n[2] Loading VALIDATION data")
#     X_val, y_val = load_split("dataset/validation")

#     return X_train, y_train, X_val, y_val

# load_data.py
import os
import scipy.io as sio
import numpy as np
from channel_config import EMOTION_CHANNEL_INDEX

def load_dataset(base_path, label_map):
    X, y = [], []

    for class_name, label in label_map.items():
        class_dir = os.path.join(base_path, class_name)

        for file in os.listdir(class_dir):
            if not file.endswith(".mat"):
                continue

            mat_path = os.path.join(class_dir, file)
            mat = sio.loadmat(mat_path)

            # Ambil variabel EEG utama (asumsi hanya 1 matrix 2D)
            eeg = next(v for v in mat.values()
                       if isinstance(v, np.ndarray) and v.ndim == 2)

            # Pilih channel emosi
            eeg_selected = eeg[EMOTION_CHANNEL_INDEX, :]
            X.append(eeg_selected)
            y.append(label)

    return X, np.array(y)
