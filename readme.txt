====================================================
README
SISTEM DETEKSI EMOSI BERBASIS EEG MENGGUNAKAN SVM
====================================================

1. DESKRIPSI UMUM
----------------------------------------------------
Proyek ini merupakan sistem deteksi emosi berbasis sinyal
Electroencephalography (EEG) menggunakan metode klasifikasi
Support Vector Machine (SVM).

Sistem ini dirancang untuk mengklasifikasikan emosi ke dalam
tiga kelas, yaitu:
- Positive
- Neutral
- Negative

Data EEG yang digunakan memiliki 62 channel, namun sistem
ini hanya memanfaatkan channel yang relevan dengan pemrosesan
emosi, yaitu area frontal, temporal, dan parietal, guna
meningkatkan performa klasifikasi dan mengurangi kompleksitas
model.


2. STRUKTUR FOLDER
----------------------------------------------------
Struktur direktori proyek adalah sebagai berikut:

Program/
│
├── dataset/
│   ├── raw/                # Data training
│   │   ├── Positive/
│   │   ├── Neutral/
│   │   └── Negative/
│   │
│   └── validation/         # Data validasi
│       ├── Positive/
│       ├── Neutral/
│       └── Negative/
│
├── models/                 # Model SVM tersimpan (.pkl)
│
├── result/
│   └── figures/            # Hasil visualisasi
│       └── svm_confusion.png
│
├── src/                    # Source code utama
│   ├── channel_config.py
│   ├── load_data.py
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   ├── model.py
│   ├── visualization.py
│   └── main.py
│
└── venv/                   # Virtual environment (opsional)


3. DESKRIPSI DATASET
----------------------------------------------------
Dataset disusun berdasarkan kelas emosi, dengan format file
.mat pada setiap folder kelas.

Untuk setiap kelas:
- 7 file EEG digunakan sebagai data training (raw)
- 3 file EEG digunakan sebagai data validasi (validation)

Setiap file .mat berisi sinyal EEG dengan bentuk:
- Dimensi: (62, N)
- 62 channel EEG
- N adalah jumlah sampel waktu (bervariasi antar file)


4. CHANNEL EEG YANG DIGUNAKAN
----------------------------------------------------
Untuk deteksi emosi, sistem ini menggunakan 14 channel EEG
yang berfokus pada area frontal, temporal, dan parietal:

Fp1, Fp2,
F3, F4, F7, F8, Fz,
T7, T8, TP7, TP8,
P3, P4, Pz

Pemilihan channel ini didasarkan pada peran neurofisiologis
area tersebut dalam pemrosesan emosi dan regulasi afektif.


5. ALUR PIPELINE SISTEM
----------------------------------------------------
Alur pemrosesan sistem secara umum adalah sebagai berikut:

1) Load Data
   - Membaca file EEG (.mat) dari folder raw dan validation
   - Label kelas ditentukan berdasarkan nama folder

2) Channel Selection
   - Memilih channel EEG yang relevan untuk deteksi emosi
   - Mengurangi dimensi data dari 62 channel menjadi 14 channel

3) Preprocessing EEG
   - Bandpass filtering (0.5–40 Hz)
   - Normalisasi sinyal menggunakan z-score

4) Feature Extraction
   - Ekstraksi fitur statistik dari setiap channel:
     * Mean
     * Standard deviation
     * Variance
     * RMS
     * Peak-to-peak
   - Total fitur: 14 channel × 5 fitur = 70 fitur per sampel

5) Training Model
   - Melatih model Support Vector Machine (SVM) dengan kernel
     linear menggunakan data training

6) Validation & Evaluation
   - Prediksi kelas emosi pada data validasi
   - Evaluasi performa menggunakan:
     * Accuracy
     * Precision
     * Recall
     * F1-score
     * Confusion Matrix

7) Output
   - Model tersimpan dalam format .pkl
   - Confusion matrix disimpan sebagai file gambar (.png)
   - Metrik evaluasi ditampilkan di terminal


6. CARA MENJALANKAN PROGRAM
----------------------------------------------------
1) Pastikan seluruh dependency telah terinstall:
   - numpy
   - scipy
   - scikit-learn
   - matplotlib
   - joblib

2) Pastikan struktur folder dataset sudah sesuai.

3) Jalankan program dari folder src:
   
   python main.py

4) Setelah program selesai dijalankan:
   - Model SVM tersimpan di folder models/
   - Confusion matrix tersimpan di folder result/figures/
   - Nilai evaluasi tampil di terminal


7. OUTPUT YANG DIHASILKAN
----------------------------------------------------
- models/svm_emotion.pkl
  Model SVM terlatih untuk deteksi emosi

- result/figures/svm_confusion.png
  Visualisasi confusion matrix hasil validasi

- Terminal output:
  Accuracy, Precision, Recall, F1-score, dan classification
  report untuk data validasi


8. CATATAN PENTING
----------------------------------------------------
- Confusion matrix dan metrik evaluasi hanya dihitung dari
  data validation untuk menghindari bias dan overfitting.
- Sistem ini dirancang untuk keperluan penelitian dan tugas
  akhir (TA).
- Penggunaan channel terpilih bertujuan meningkatkan
  generalisasi model dan interpretabilitas hasil.


====================================================
END OF README
====================================================
