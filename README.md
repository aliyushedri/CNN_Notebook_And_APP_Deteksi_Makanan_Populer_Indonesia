# 🍛 Klasifikasi Makanan Indonesia dengan CNN

Proyek deep learning untuk mengklasifikasikan gambar makanan khas Indonesia menggunakan
**Convolutional Neural Network (CNN)**, dilengkapi aplikasi GUI untuk deteksi gambar.

**Dataset:** [Indonesian Food Dataset — Kaggle](https://www.kaggle.com/datasets/rizkyyk/dataset-food-classification)

## 📁 Struktur Proyek

```text
CNN_FOOD_INDONESIAN/
├── training_cnn_food_indonesian.ipynb   # Notebook training CNN (download dataset via Kaggle API)
├── app/
│   └── food_detector_app.py             # Aplikasi GUI deteksi makanan (Tkinter)
├── dataset/                             # Dataset dari Kaggle (dibuat otomatis oleh notebook)
├── models/                              # Model terlatih + metadata (dibuat otomatis)
│   ├── cnn_food_indonesian_best.keras   #   - checkpoint bobot terbaik
│   ├── cnn_food_indonesian_final.keras  #   - model final
│   ├── class_names.json                 #   - daftar nama kelas
│   └── model_metadata.json              #   - info model (akurasi, tanggal, dll.)
├── outputs/                             # Grafik hasil training & evaluasi (dibuat otomatis)
├── requirements.txt                     # Daftar dependensi Python
└── README.md
```

## 🚀 Cara Menjalankan

### 1. Instal dependensi

```bash
pip install -r requirements.txt
```

### 2. Siapkan kredensial Kaggle

1. Login ke [kaggle.com](https://www.kaggle.com) → foto profil → **Settings** → bagian **API** → **Create New Token**.
2. Letakkan file `kaggle.json` yang terunduh di:
   - **Windows:** `C:\Users\<nama_user>\.kaggle\kaggle.json`
   - **Linux/Mac:** `~/.kaggle/kaggle.json`

### 3. Jalankan notebook training

Buka `training_cnn_food_indonesian.ipynb` (Jupyter / VS Code) dan jalankan sel dari atas
ke bawah. Notebook akan otomatis:

- Membuat struktur folder (`dataset/`, `models/`, `outputs/`).
- Mengunduh dataset dari Kaggle melalui Kaggle API.
- Melatih CNN dan menyimpan model + metadata ke `models/`.

### 4. Jalankan aplikasi UI deteksi

```bash
python app/food_detector_app.py
```

**Fitur aplikasi:**

| Fitur | Keterangan |
|---|---|
| 🔄 Select Model | Pilih model dari folder `models/` atau browse file `.keras`/`.h5` |
| ⚙️ Load Model | Muat model di background (UI tetap responsif) |
| ℹ️ Info Model | Lihat akurasi test, jumlah kelas, tanggal training |
| 🖼️ Buka Gambar | Pilih gambar makanan + preview |
| 🔍 Detect | Prediksi kelas + bar chart Top-K probabilitas |
| 📜 Riwayat | Tabel riwayat semua hasil deteksi |
| 💾 Ekspor CSV | Simpan riwayat deteksi ke file CSV |

## 🧠 Arsitektur Model

CNN *from scratch* dengan komposisi: augmentasi data (flip, rotasi, zoom, kontras) →
rescaling 1/255 → 4 blok konvolusi (32→64→128→256 filter, masing-masing
Conv2D + BatchNorm + ReLU + MaxPool + Dropout) → GlobalAveragePooling →
Dense 256 → Softmax.

Training memakai optimizer **Adam**, loss **sparse categorical crossentropy**, serta
callback **ModelCheckpoint**, **EarlyStopping**, dan **ReduceLROnPlateau**.
