# import numpy as np
# from scipy.signal import welch

# BANDS = [
#     (0.5, 4), (4, 8), (8, 13), (13, 30), (30, 45)
# ]

# def bandpower(signal, fs, band):
#     f, psd = welch(signal, fs)
#     idx = (f >= band[0]) & (f <= band[1])
#     return np.sum(psd[idx])

# def extract_features(X, fs, name="DATA"):
#     print(f"[Feature Extraction] {name}")
#     features = []

#     for trial in X:
#         trial_feat = []
#         for ch in trial:
#             for band in BANDS:
#                 trial_feat.append(bandpower(ch, fs, band))
#         features.append(trial_feat)

#     features = np.array(features)
#     print(f"  Feature shape: {features.shape}")
#     return features

# feature_extraction.py
import numpy as np

def extract_features(eeg):
    features = []

    for ch in eeg:
        features.extend([
            np.mean(ch),
            np.std(ch),
            np.var(ch),
            np.sqrt(np.mean(ch**2)),   # RMS
            np.max(ch) - np.min(ch)    # Peak-to-peak
        ])

    return np.array(features)
