import numpy as np

def extract_features(X):
    features = []

    for eeg in X:
        mean = np.mean(eeg, axis=1)
        std  = np.std(eeg, axis=1)
        rms  = np.sqrt(np.mean(eeg**2, axis=1))

        feat = np.concatenate([mean, std, rms])
        features.append(feat)

    return np.array(features)
