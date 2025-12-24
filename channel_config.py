# channel_config.py

CHANNEL_NAMES = [
    "Fp1","Fp2","F3","F4","C3","C4","P3","P4","O1","O2",
    "F7","F8","T7","T8","P7","P8","Fz","Cz","Pz",
    "TP7","TP8"
]

# Channel yang dipakai untuk emotion detection
EMOTION_CHANNELS = [
    "Fp1","Fp2","F3","F4","F7","F8","Fz",
    "T7","T8","TP7","TP8",
    "P3","P4","Pz"
]

# Ambil index channel
EMOTION_CHANNEL_INDEX = [
    CHANNEL_NAMES.index(ch) for ch in EMOTION_CHANNELS
]
