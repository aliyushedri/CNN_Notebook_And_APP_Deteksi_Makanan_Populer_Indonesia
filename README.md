# Klasifikasi Makanan Indonesia dengan EfficientNetB0

Proyek deep learning untuk mengklasifikasikan 13 kelas makanan menggunakan
**Convolutional Neural Network (CNN) berbasis EfficientNetB0**. Model dilatih dengan
transfer learning dan fine-tuning, lalu digunakan oleh aplikasi desktop Tkinter untuk
melakukan prediksi gambar makanan.

**Dataset:** [Indonesian Food Dataset - Kaggle](https://www.kaggle.com/datasets/rizkyyk/dataset-food-classification)

## Struktur Proyek

```text
CNN_FOOD_INDONESIAN/
|-- training_cnn_food_indonesian.ipynb   # Notebook training dan evaluasi model
|-- app/
|   `-- food_detector_app.py             # Aplikasi desktop deteksi makanan
|-- models/
|   |-- cnn_food_indonesian_final.keras  # Model final siap pakai
|   |-- class_names.json                 # Urutan nama kelas
|   `-- model_metadata.json              # Metadata model dan metrik test
|-- outputs/                             # Kurva training dan confusion matrix
|-- requirements.txt                     # Dependensi proyek
`-- README.md
```

Folder `dataset/` tidak disimpan di GitHub karena berukuran besar. Notebook akan
mengunduhnya kembali melalui Kaggle API bila dataset belum tersedia.

## Menjalankan Proyek

### 1. Instal dependensi

```bash
pip install -r requirements.txt
```

### 2. Siapkan kredensial Kaggle

1. Login ke Kaggle, lalu buka **Settings** > **API** > **Create New Token**.
2. Letakkan `kaggle.json` pada lokasi berikut:

   - Windows: `C:\Users\<nama_user>\.kaggle\kaggle.json`
   - Linux/macOS: `~/.kaggle/kaggle.json`

### 3. Training dan evaluasi model

Buka `training_cnn_food_indonesian.ipynb` di Jupyter atau VS Code, lalu jalankan sel
secara berurutan. Notebook akan membuat struktur folder, mengunduh dataset, melatih
model, membuat grafik evaluasi, dan menyimpan model final beserta metadatanya.

### 4. Jalankan aplikasi deteksi

```bash
python app/food_detector_app.py
```

## Fitur Aplikasi

| Fitur | Keterangan |
|---|---|
| Pilih model | Memilih model dari folder `models/` atau lokasi lain. |
| Muat model | Memuat model pada thread terpisah agar antarmuka tetap responsif. |
| Info model | Menampilkan jumlah kelas, ukuran input, parameter, dan metadata evaluasi. |
| Pilih gambar | Menampilkan pratinjau gambar yang akan diklasifikasikan. |
| Deteksi gambar | Menampilkan prediksi Top-K dengan confidence dua desimal. |
| Riwayat deteksi | Menyimpan hasil prediksi selama aplikasi berjalan. |
| Ekspor CSV | Mengekspor riwayat deteksi ke berkas CSV. |

## Arsitektur dan Training

Model memakai **EfficientNetB0**, yaitu arsitektur CNN yang telah dipretrain pada
ImageNet. Konfigurasi training terdiri dari:

1. Augmentasi data: horizontal flip, rotasi, zoom, dan perubahan kontras.
2. Transfer learning: backbone EfficientNetB0 dibekukan untuk melatih classifier head.
3. Fine-tuning: 30 layer terakhir backbone dilatih ulang dengan learning rate kecil.
4. Evaluasi: accuracy, loss, classification report, dan confusion matrix pada data test.

Metrik model final yang tersimpan saat ini adalah **94.00% test accuracy** dan
**94.00% macro F1-score** pada 650 citra data test.
