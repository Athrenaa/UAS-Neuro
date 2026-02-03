# # import os
# # import librosa
# # import numpy as np
# # import soundfile as sf
# # from scipy.signal import find_peaks

# # # =========================
# # # CONFIG
# # # =========================
# # BASE_INPUT_DIR  = "data/b4split"
# # BASE_OUTPUT_DIR = "data/split"

# # CLASSES = ["cough", "covid", "noncough", "tuberculosis"]

# # SAMPLE_RATE = 16000
# # SEGMENT_DURATION = 0.5  # seconds
# # SEGMENT_SAMPLES = int(SAMPLE_RATE * SEGMENT_DURATION)

# # FRAME_LENGTH = int(0.025 * SAMPLE_RATE)   # 25 ms
# # HOP_LENGTH   = int(0.010 * SAMPLE_RATE)   # 10 ms

# # THRESHOLD_K = 1.5
# # MIN_PEAK_DISTANCE = int(0.3 * SAMPLE_RATE / HOP_LENGTH)  # 300 ms

# # # =========================
# # # SEGMENT FUNCTION
# # # =========================
# # def segment_cough(audio_path, output_dir, class_name, sr=SAMPLE_RATE):
# #     os.makedirs(output_dir, exist_ok=True)

# #     # Load audio
# #     y, _ = librosa.load(audio_path, sr=sr)
# #     y = librosa.util.normalize(y)

# #     # RMS energy
# #     rms = librosa.feature.rms(
# #         y=y,
# #         frame_length=FRAME_LENGTH,
# #         hop_length=HOP_LENGTH
# #     )[0]

# #     # Adaptive threshold
# #     threshold = np.mean(rms) + THRESHOLD_K * np.std(rms)

# #     # Peak detection
# #     peaks, _ = find_peaks(
# #         rms,
# #         height=threshold,
# #         distance=MIN_PEAK_DISTANCE
# #     )

# #     half_seg = SEGMENT_SAMPLES // 2
# #     count = 0

# #     for peak in peaks:
# #         center = peak * HOP_LENGTH
# #         start  = center - half_seg
# #         end    = center + half_seg

# #         if start < 0 or end > len(y):
# #             continue

# #         segment = y[start:end]

# #         out_name = f"{class_name}_{count:05d}.wav"
# #         out_path = os.path.join(output_dir, out_name)

# #         sf.write(out_path, segment, sr)
# #         count += 1

# #     return count

# # # =========================
# # # PROCESS ALL CLASSES
# # # =========================
# # def process_all_classes():
# #     print("=== START SEGMENTATION ===")

# #     for cls in CLASSES:
# #         input_dir  = os.path.join(BASE_INPUT_DIR, cls)
# #         output_dir = os.path.join(BASE_OUTPUT_DIR, cls)

# #         if not os.path.exists(input_dir):
# #             print(f"[SKIP] Folder tidak ditemukan: {input_dir}")
# #             continue

# #         total_segments = 0
# #         files = [f for f in os.listdir(input_dir) if f.endswith(".wav")]

# #         print(f"\n[CLASS] {cls} | {len(files)} file")

# #         for file in files:
# #             in_path = os.path.join(input_dir, file)
# #             n_seg = segment_cough(
# #                 audio_path=in_path,
# #                 output_dir=output_dir,
# #                 class_name=cls
# #             )
# #             total_segments += n_seg

# #         print(f"[DONE] {cls}: {total_segments} segmen disimpan")

# #     print("\n=== SEGMENTATION SELESAI ===")

# # # =========================
# # # MAIN
# # # =========================
# # if __name__ == "__main__":
# #     process_all_classes()

# import os
# import librosa
# import numpy as np
# import soundfile as sf
# from scipy.signal import find_peaks

# # =========================
# # CONFIG
# # =========================
# BASE_INPUT_DIR  = "data/b4split"
# BASE_OUTPUT_DIR = "data/split"

# CLASSES = ["cough", "covid", "noncough", "tuberculosis"]

# SAMPLE_RATE = 16000
# SEGMENT_DURATION = 0.5
# SEGMENT_SAMPLES = int(SAMPLE_RATE * SEGMENT_DURATION)

# FRAME_LENGTH = int(0.025 * SAMPLE_RATE)
# HOP_LENGTH   = int(0.010 * SAMPLE_RATE)

# THRESHOLD_K = 1.5
# MIN_PEAK_DISTANCE = int(0.3 * SAMPLE_RATE / HOP_LENGTH)

# # =========================
# # SEGMENT FUNCTION
# # =========================
# def segment_cough(audio_path, output_dir, class_name, start_idx, sr=SAMPLE_RATE):
#     y, _ = librosa.load(audio_path, sr=sr)
#     y = librosa.util.normalize(y)

#     rms = librosa.feature.rms(
#         y=y,
#         frame_length=FRAME_LENGTH,
#         hop_length=HOP_LENGTH
#     )[0]

#     threshold = np.mean(rms) + THRESHOLD_K * np.std(rms)

#     peaks, _ = find_peaks(
#         rms,
#         height=threshold,
#         distance=MIN_PEAK_DISTANCE
#     )

#     half_seg = SEGMENT_SAMPLES // 2
#     idx = start_idx

#     for peak in peaks:
#         center = peak * HOP_LENGTH
#         start  = center - half_seg
#         end    = center + half_seg

#         if start < 0 or end > len(y):
#             continue

#         segment = y[start:end]
#         out_name = f"{class_name}_{idx:06d}.wav"
#         sf.write(os.path.join(output_dir, out_name), segment, sr)
#         idx += 1

#     return idx - start_idx  # jumlah segmen baru
# def process_all_classes():
#     print("=== START SEGMENTATION ===")

#     for cls in CLASSES:
#         input_dir  = os.path.join(BASE_INPUT_DIR, cls)
#         output_dir = os.path.join(BASE_OUTPUT_DIR, cls)
#         os.makedirs(output_dir, exist_ok=True)

#         files = [f for f in os.listdir(input_dir) if f.endswith(".wav")]
#         global_counter = 0

#         print(f"\n[CLASS] {cls} | {len(files)} file")

#         for f in files:
#             path = os.path.join(input_dir, f)
#             n_seg = segment_cough(
#                 audio_path=path,
#                 output_dir=output_dir,
#                 class_name=cls,
#                 start_idx=global_counter
#             )
#             global_counter += n_seg

#         print(f"[DONE] {cls}: {global_counter} segmen disimpan")

#     print("\n=== SEGMENTATION SELESAI ===")

# if __name__ == "__main__":
#     process_all_classes()
