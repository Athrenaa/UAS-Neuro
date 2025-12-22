import numpy as np
from scipy.signal import butter, filtfilt

def bandpass_filter(eeg, fs=250, low=1, high=40, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype="band")
    return filtfilt(b, a, eeg, axis=1)

def preprocess_eeg(X):
    X_out = []
    for eeg in X:
        eeg_filt = bandpass_filter(eeg)
        X_out.append(eeg_filt)
    return np.array(X_out)
