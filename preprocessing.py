# from scipy.signal import butter, filtfilt
# import numpy as np

# def bandpass(signal, fs, low=0.5, high=45):
#     nyq = fs / 2
#     b, a = butter(4, [low/nyq, high/nyq], btype="band")
#     return filtfilt(b, a, signal)

# def preprocess(X, fs, name="DATA"):
#     print(f"[Preprocessing] {name}")
#     X_out = []

#     for i, trial in enumerate(X):
#         filtered = []
#         for ch in trial:
#             filtered.append(bandpass(ch, fs))
#         X_out.append(filtered)

#         if (i + 1) % 10 == 0 or (i + 1) == len(X):
#             print(f"  Processed {i+1}/{len(X)}")

#     return np.array(X_out)

# preprocessing.py
import numpy as np
from scipy.signal import butter, filtfilt

def bandpass_filter(signal, fs=250, low=0.5, high=40):
    b, a = butter(4, [low/(fs/2), high/(fs/2)], btype="band")
    return filtfilt(b, a, signal)

def preprocess_eeg(eeg):
    processed = []

    for ch in eeg:
        ch = bandpass_filter(ch)
        ch = (ch - np.mean(ch)) / np.std(ch)  # z-score
        processed.append(ch)

    return np.array(processed)
