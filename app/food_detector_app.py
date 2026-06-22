"""
============================================================
 APLIKASI DETEKSI MAKANAN INDONESIA (GUI)
------------------------------------------------------------
 Aplikasi desktop berbasis Tkinter untuk mendeteksi jenis
 makanan Indonesia dari sebuah gambar menggunakan model CNN
 yang dilatih pada notebook training_cnn_food_indonesian.ipynb.

 Fitur:
   1. Select Model  - memilih file model (.keras/.h5) dari
                      folder models/ atau browse manual.
   2. Load Model    - memuat model di thread terpisah agar
                      UI tidak macet (freeze).
   3. Buka Gambar   - memilih gambar & menampilkan preview.
   4. Detect        - memprediksi kelas makanan + menampilkan
                      Top-K probabilitas sebagai bar chart.
   5. Riwayat       - mencatat semua hasil deteksi dalam tabel.
   6. Ekspor CSV    - menyimpan riwayat deteksi ke file CSV.
   7. Info Model    - menampilkan metadata model (akurasi test,
                      jumlah kelas, tanggal training).

 Cara menjalankan (dari folder proyek):
   python app/food_detector_app.py
============================================================
"""

import csv
import json
import threading
from datetime import datetime
from pathlib import Path
from tkinter import Tk, StringVar, Canvas, filedialog, messagebox
from tkinter import ttk

import numpy as np
from PIL import Image, ImageTk

# ------------------------------------------------------------
# Konstanta & path penting
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent   # folder proyek (CNN_FOOD_INDONESIAN)
MODELS_DIR = BASE_DIR / "models"                    # lokasi default model terlatih
IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
PREVIEW_SIZE = (420, 420)                           # ukuran maksimum preview gambar
TOP_K_DEFAULT = 5                                   # jumlah prediksi teratas ditampilkan

# Warna bar hasil prediksi: hijau utk peringkat 1, biru utk sisanya
WARNA_BAR_UTAMA = "#2e9e5b"
WARNA_BAR_LAIN = "#4a7fb5"


class FoodDetectorApp:
    """Kelas utama aplikasi GUI deteksi makanan Indonesia."""

    # ==========================================================
    # INISIALISASI & PENYUSUNAN LAYOUT
    # ==========================================================
    def __init__(self, root: Tk):
        self.root = root
        self.root.title("🍛 Deteksi Makanan Indonesia — CNN")
        self.root.geometry("1280x780")
        self.root.minsize(1080, 680)
        self.root.title("Deteksi Makanan Indonesia | AI Vision")

        # --- State aplikasi -----------------------------------
        self.model = None              # objek model Keras yang sudah dimuat
        self.class_names = []          # daftar nama kelas sesuai urutan output model
        self.input_size = (224, 224)   # ukuran input model (dibaca dari model)
        self.image_path = None         # path gambar yang sedang dibuka
        self.image_pil = None          # objek PIL gambar asli
        self.metadata = {}             # isi model_metadata.json (jika ada)
        self.riwayat = []              # daftar dict hasil deteksi untuk ekspor CSV

        # --- Variabel yang terikat ke widget ------------------
        self.var_model = StringVar()       # path model terpilih di combobox
        self.var_status = StringVar(value="Siap. Pilih model lalu klik 'Load Model'.")
        self.var_topk = StringVar(value=str(TOP_K_DEFAULT))

        self.var_model_detail = StringVar(value="Belum ada model yang dipilih.")
        self._model_paths = {}

        self._configure_styles()
        self._susun_layout()
        self._refresh_daftar_model()

    def _susun_layout(self):
        """Menyusun seluruh widget: panel kontrol kiri, preview tengah, hasil kanan."""
        # --- Frame utama (3 kolom) -----------------------------
        utama = ttk.Frame(self.root, padding=10)
        utama.pack(fill="both", expand=True)
        utama.columnconfigure(1, weight=3)   # kolom preview melar
        utama.columnconfigure(2, weight=2)   # kolom hasil melar
        utama.rowconfigure(0, weight=1)

        # ====================== PANEL KIRI: KONTROL ======================
        kiri = ttk.LabelFrame(utama, text=" Kontrol ", padding=10)
        kiri.grid(row=0, column=0, sticky="nsw", padx=(0, 8))

        # -- 1) Pemilihan model --------------------------------
        ttk.Label(kiri, text="Model CNN:").pack(anchor="w")
        self.combo_model = ttk.Combobox(
            kiri, textvariable=self.var_model, width=34, state="readonly"
        )
        self.combo_model.pack(fill="x", pady=(2, 4))

        baris_model = ttk.Frame(kiri)
        baris_model.pack(fill="x")
        ttk.Button(baris_model, text="🔄 Refresh", command=self._refresh_daftar_model)\
            .pack(side="left", expand=True, fill="x", padx=(0, 2))
        ttk.Button(baris_model, text="📂 Browse...", command=self._browse_model)\
            .pack(side="left", expand=True, fill="x", padx=(2, 0))

        self.btn_load = ttk.Button(kiri, text="⚙️ Load Model", command=self._load_model)
        self.btn_load.pack(fill="x", pady=(6, 2))

        ttk.Button(kiri, text="ℹ️ Info Model", command=self._tampilkan_info_model)\
            .pack(fill="x", pady=(0, 10))

        ttk.Separator(kiri).pack(fill="x", pady=4)

        # -- 2) Pemilihan gambar & deteksi ----------------------
        ttk.Button(kiri, text="🖼️ Buka Gambar...", command=self._buka_gambar)\
            .pack(fill="x", pady=(8, 2))

        baris_topk = ttk.Frame(kiri)
        baris_topk.pack(fill="x", pady=(4, 2))
        ttk.Label(baris_topk, text="Top-K prediksi:").pack(side="left")
        ttk.Spinbox(baris_topk, from_=1, to=13, width=5, textvariable=self.var_topk)\
            .pack(side="right")

        self.btn_detect = ttk.Button(
            kiri, text="🔍 DETECT", command=self._deteksi, state="disabled"
        )
        self.btn_detect.pack(fill="x", pady=(6, 10), ipady=6)

        ttk.Separator(kiri).pack(fill="x", pady=4)

        # -- 3) Utilitas riwayat --------------------------------
        ttk.Button(kiri, text="💾 Ekspor Riwayat (CSV)", command=self._ekspor_csv)\
            .pack(fill="x", pady=(8, 2))
        ttk.Button(kiri, text="🗑️ Bersihkan Riwayat", command=self._bersihkan_riwayat)\
            .pack(fill="x", pady=(0, 2))

        # ====================== PANEL TENGAH: PREVIEW ======================
        tengah = ttk.LabelFrame(utama, text=" Preview Gambar ", padding=10)
        tengah.grid(row=0, column=1, sticky="nsew", padx=(0, 8))
        tengah.rowconfigure(0, weight=1)
        tengah.columnconfigure(0, weight=1)

        self.lbl_preview = ttk.Label(
            tengah, text="\n\nBelum ada gambar.\nKlik 'Buka Gambar...' untuk memilih.",
            anchor="center", justify="center"
        )
        self.lbl_preview.grid(row=0, column=0, sticky="nsew")

        self.lbl_nama_file = ttk.Label(tengah, text="", anchor="center")
        self.lbl_nama_file.grid(row=1, column=0, sticky="ew", pady=(6, 0))

        # ====================== PANEL KANAN: HASIL ======================
        kanan = ttk.Frame(utama)
        kanan.grid(row=0, column=2, sticky="nsew")
        kanan.rowconfigure(0, weight=3)
        kanan.rowconfigure(1, weight=2)
        kanan.columnconfigure(0, weight=1)

        # -- Hasil prediksi (bar chart digambar di Canvas) -------
        frame_hasil = ttk.LabelFrame(kanan, text=" Hasil Deteksi ", padding=6)
        frame_hasil.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
        frame_hasil.rowconfigure(0, weight=1)
        frame_hasil.columnconfigure(0, weight=1)

        self.canvas_hasil = Canvas(frame_hasil, highlightthickness=0)
        self.canvas_hasil.grid(row=0, column=0, sticky="nsew")

        # -- Tabel riwayat deteksi --------------------------------
        frame_riwayat = ttk.LabelFrame(kanan, text=" Riwayat Deteksi ", padding=6)
        frame_riwayat.grid(row=1, column=0, sticky="nsew")
        frame_riwayat.rowconfigure(0, weight=1)
        frame_riwayat.columnconfigure(0, weight=1)

        kolom = ("waktu", "file", "prediksi", "confidence")
        self.tabel_riwayat = ttk.Treeview(
            frame_riwayat, columns=kolom, show="headings", height=6
        )
        for nama, lebar in zip(kolom, (70, 130, 110, 80)):
            self.tabel_riwayat.heading(nama, text=nama.capitalize())
            self.tabel_riwayat.column(nama, width=lebar, anchor="w")
        self.tabel_riwayat.grid(row=0, column=0, sticky="nsew")

        scroll = ttk.Scrollbar(frame_riwayat, command=self.tabel_riwayat.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.tabel_riwayat.configure(yscrollcommand=scroll.set)

        # ====================== STATUS BAR BAWAH ======================
        status = ttk.Frame(self.root, padding=(10, 4))
        status.pack(fill="x", side="bottom")
        ttk.Label(status, textvariable=self.var_status, anchor="w").pack(side="left")
        self.progress = ttk.Progressbar(status, mode="indeterminate", length=160)
        self.progress.pack(side="right")

    # ==========================================================
    # FITUR 1 & 2: SELECT MODEL + LOAD MODEL
    # ==========================================================
    def _refresh_daftar_model(self):
        """Memindai folder models/ dan mengisi combobox dengan file model."""
        daftar = []
        if MODELS_DIR.exists():
            daftar = sorted(
                str(p) for p in MODELS_DIR.iterdir()
                if p.suffix.lower() in (".keras", ".h5")
            )
        self.combo_model["values"] = daftar
        if daftar and not self.var_model.get():
            self.combo_model.current(0)  # otomatis pilih model pertama
        self._set_status(f"Ditemukan {len(daftar)} model di folder models/.")

    def _browse_model(self):
        """Membuka dialog untuk memilih file model di lokasi mana pun."""
        path = filedialog.askopenfilename(
            title="Pilih file model",
            filetypes=[("Model Keras", "*.keras *.h5"), ("Semua file", "*.*")],
            initialdir=MODELS_DIR if MODELS_DIR.exists() else BASE_DIR,
        )
        if path:
            self.var_model.set(path)

    def _load_model(self):
        """Memuat model di thread terpisah agar jendela UI tetap responsif."""
        path = self.var_model.get()
        if not path:
            messagebox.showwarning("Model belum dipilih",
                                   "Pilih file model terlebih dahulu.")
            return

        # Nonaktifkan tombol & nyalakan progress bar selama proses load
        self.btn_load.config(state="disabled")
        self.btn_detect.config(state="disabled")
        self.progress.start(12)
        self._set_status(f"Memuat model: {Path(path).name} ...")

        threading.Thread(target=self._load_model_worker, args=(path,),
                         daemon=True).start()

    def _load_model_worker(self, path: str):
        """Worker thread: memuat model Keras + metadata kelas dari disk."""
        try:
            # Import tensorflow di sini agar aplikasi terbuka cepat
            # (import TF berat, hanya dibutuhkan saat load model)
            from tensorflow import keras

            model = keras.models.load_model(path)

            # Baca ukuran input dari model: (None, tinggi, lebar, 3)
            shape = model.input_shape
            input_size = (shape[1] or 224, shape[2] or 224)

            # Cari daftar nama kelas di folder yang sama dengan model
            class_names, metadata = self._muat_metadata(Path(path).parent, model)

            # Kirim hasil kembali ke thread UI (Tkinter tidak thread-safe)
            self.root.after(0, self._load_model_selesai,
                            model, class_names, input_size, metadata, path, None)
        except Exception as exc:  # tampilkan error apa pun ke pengguna
            self.root.after(0, self._load_model_selesai,
                            None, None, None, None, path, exc)

    @staticmethod
    def _muat_metadata(folder: Path, model):
        """Membaca class_names.json & model_metadata.json di samping file model.

        Jika file tidak ditemukan, nama kelas di-generate generik
        ('Kelas_0', 'Kelas_1', ...) sesuai jumlah neuron output model.
        """
        metadata = {}
        file_meta = folder / "model_metadata.json"
        if file_meta.exists():
            metadata = json.loads(file_meta.read_text(encoding="utf-8"))

        file_kelas = folder / "class_names.json"
        if file_kelas.exists():
            class_names = json.loads(file_kelas.read_text(encoding="utf-8"))
        elif "nama_kelas" in metadata:
            class_names = metadata["nama_kelas"]
        else:
            jumlah_output = model.output_shape[-1]
            class_names = [f"Kelas_{i}" for i in range(jumlah_output)]
        return class_names, metadata

    def _load_model_selesai(self, model, class_names, input_size,
                            metadata, path, error):
        """Callback di thread UI setelah proses load model selesai/gagal."""
        self.progress.stop()
        self.btn_load.config(state="normal")

        if error is not None:
            messagebox.showerror("Gagal memuat model", str(error))
            self._set_status("Gagal memuat model.")
            return

        self.model = model
        self.class_names = class_names
        self.input_size = input_size
        self.metadata = metadata
        self.btn_detect.config(state="normal")  # deteksi kini diizinkan
        self._set_status(
            f"✅ Model '{Path(path).name}' dimuat — "
            f"{len(class_names)} kelas, input {input_size[0]}x{input_size[1]}."
        )

    def _tampilkan_info_model(self):
        """Menampilkan metadata model (akurasi, kelas, tanggal training)."""
        if self.model is None:
            messagebox.showinfo("Info Model", "Belum ada model yang dimuat.")
            return
        info = [
            f"Jumlah kelas : {len(self.class_names)}",
            f"Ukuran input : {self.input_size[0]} x {self.input_size[1]}",
            f"Jumlah parameter : {self.model.count_params():,}",
        ]
        # Tambahkan info dari model_metadata.json jika tersedia
        if self.metadata:
            info += [
                f"Akurasi test : {self.metadata.get('akurasi_test', '-')}",
                f"Tanggal training : {self.metadata.get('tanggal_training', '-')}",
                f"Dataset : {self.metadata.get('dataset_kaggle', '-')}",
            ]
        info.append("\nDaftar kelas:\n" + ", ".join(self.class_names))
        messagebox.showinfo("Info Model", "\n".join(info))

    # ==========================================================
    # FITUR 3: BUKA GAMBAR & PREVIEW
    # ==========================================================
    def _buka_gambar(self):
        """Memilih file gambar lalu menampilkannya pada panel preview."""
        path = filedialog.askopenfilename(
            title="Pilih gambar makanan",
            filetypes=[("Gambar", " ".join(IMG_EXTS)), ("Semua file", "*.*")],
        )
        if not path:
            return

        try:
            self.image_pil = Image.open(path).convert("RGB")
        except Exception as exc:
            messagebox.showerror("Gagal membuka gambar", str(exc))
            return

        self.image_path = Path(path)

        # Buat thumbnail agar muat di panel preview tanpa merusak aspect ratio
        thumb = self.image_pil.copy()
        thumb.thumbnail(PREVIEW_SIZE)
        self._foto_preview = ImageTk.PhotoImage(thumb)  # simpan referensi (wajib di Tk)
        self.lbl_preview.config(image=self._foto_preview, text="")
        self.lbl_nama_file.config(
            text=f"{self.image_path.name}  ({self.image_pil.width}x{self.image_pil.height})"
        )
        self._set_status(f"Gambar dibuka: {self.image_path.name}")

    # ==========================================================
    # FITUR 4: DETEKSI / PREDIKSI
    # ==========================================================
    def _deteksi(self):
        """Validasi input lalu menjalankan prediksi di thread terpisah."""
        if self.model is None:
            messagebox.showwarning("Model belum dimuat", "Load model terlebih dahulu.")
            return
        if self.image_pil is None:
            messagebox.showwarning("Gambar belum dipilih", "Buka gambar terlebih dahulu.")
            return

        self.btn_detect.config(state="disabled")
        self.progress.start(12)
        self._set_status("Mendeteksi ...")
        threading.Thread(target=self._deteksi_worker, daemon=True).start()

    def _deteksi_worker(self):
        """Worker thread: pra-pemrosesan gambar + prediksi model."""
        try:
            # 1) Resize gambar ke ukuran input model & ubah ke array float
            img = self.image_pil.resize(self.input_size)
            array = np.asarray(img, dtype=np.float32)
            array = np.expand_dims(array, axis=0)  # tambah dimensi batch -> (1, H, W, 3)

            # 2) Prediksi. CATATAN: normalisasi piksel (Rescaling 1/255)
            #    sudah menjadi bagian dari model, jadi cukup kirim nilai mentah.
            probabilitas = self.model.predict(array, verbose=0)[0]

            self.root.after(0, self._deteksi_selesai, probabilitas, None)
        except Exception as exc:
            self.root.after(0, self._deteksi_selesai, None, exc)

    def _deteksi_selesai(self, probabilitas, error):
        """Callback di thread UI: menampilkan hasil & mencatat ke riwayat."""
        self.progress.stop()
        self.btn_detect.config(state="normal")

        if error is not None:
            messagebox.showerror("Gagal mendeteksi", str(error))
            self._set_status("Gagal mendeteksi.")
            return

        # Ambil Top-K indeks kelas dengan probabilitas tertinggi
        try:
            top_k = max(1, min(int(self.var_topk.get()), len(self.class_names)))
        except ValueError:
            top_k = TOP_K_DEFAULT
        urutan = np.argsort(probabilitas)[::-1][:top_k]

        hasil = [(self.class_names[i], float(probabilitas[i])) for i in urutan]
        self._gambar_bar_hasil(hasil)

        # Catat hasil terbaik ke riwayat (tabel + list untuk ekspor CSV)
        nama_top, conf_top = hasil[0]
        waktu = datetime.now().strftime("%H:%M:%S")
        self.riwayat.append({
            "waktu": waktu,
            "file": self.image_path.name,
            "prediksi": nama_top,
            "confidence": f"{conf_top*100:.2f}%",
        })
        self.tabel_riwayat.insert(
            "", 0, values=(waktu, self.image_path.name, nama_top, f"{conf_top*100:.2f}%")
        )
        self._set_status(f"✅ Terdeteksi: {nama_top} ({conf_top*100:.2f}%)")

    def _gambar_bar_hasil(self, hasil):
        """Menggambar hasil Top-K sebagai bar chart horizontal di Canvas.

        Args:
            hasil: list of (nama_kelas, probabilitas) terurut menurun.
        """
        c = self.canvas_hasil
        c.delete("all")  # bersihkan hasil sebelumnya
        c.update_idletasks()

        lebar = max(c.winfo_width(), 240)
        tinggi_baris = 44
        margin = 8
        lebar_bar_max = lebar - 2 * margin

        for i, (nama, prob) in enumerate(hasil):
            y = margin + i * tinggi_baris
            warna = WARNA_BAR_UTAMA if i == 0 else WARNA_BAR_LAIN

            # Teks label: nama kelas + persentase
            c.create_text(margin, y, anchor="nw", font=("Segoe UI", 10, "bold"),
                          text=f"{i+1}. {nama}  —  {prob*100:.2f}%")

            # Bar latar (abu-abu) + bar nilai (berwarna)
            y_bar = y + 20
            c.create_rectangle(margin, y_bar, margin + lebar_bar_max, y_bar + 14,
                               fill="#e3e3e3", outline="")
            c.create_rectangle(margin, y_bar, margin + lebar_bar_max * prob, y_bar + 14,
                               fill=warna, outline="")

    # ==========================================================
    # FITUR 5 & 6: RIWAYAT + EKSPOR CSV
    # ==========================================================
    def _ekspor_csv(self):
        """Menyimpan seluruh riwayat deteksi ke file CSV pilihan pengguna."""
        if not self.riwayat:
            messagebox.showinfo("Riwayat kosong", "Belum ada hasil deteksi untuk diekspor.")
            return

        path = filedialog.asksaveasfilename(
            title="Simpan riwayat deteksi",
            defaultextension=".csv",
            initialfile=f"riwayat_deteksi_{datetime.now():%Y%m%d_%H%M%S}.csv",
            filetypes=[("File CSV", "*.csv")],
        )
        if not path:
            return

        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["waktu", "file", "prediksi", "confidence"])
            writer.writeheader()
            writer.writerows(self.riwayat)
        self._set_status(f"💾 Riwayat diekspor ke {Path(path).name}")

    def _bersihkan_riwayat(self):
        """Mengosongkan tabel riwayat dan data ekspor."""
        self.riwayat.clear()
        for item in self.tabel_riwayat.get_children():
            self.tabel_riwayat.delete(item)
        self._set_status("Riwayat dibersihkan.")

    # ==========================================================
    # UTILITAS
    # ==========================================================
    def _set_status(self, teks: str):
        """Memperbarui teks pada status bar bawah."""
        self.var_status.set(teks)


# ------------------------------------------------------------
# Titik masuk program
    # ==========================================================
    # UI PROFESIONAL: HIJAU, BIRU, PUTIH, DAN NAVY
    # ==========================================================
    def _configure_styles(self):
        """Menerapkan gaya visual yang konsisten pada seluruh aplikasi."""
        style = ttk.Style(self.root)
        style.theme_use("clam")
        self.root.configure(bg="#F3F7FA")

        style.configure("App.TFrame", background="#F3F7FA")
        style.configure("Header.TFrame", background="#102A43")
        style.configure("HeaderTitle.TLabel", background="#102A43", foreground="#FFFFFF",
                        font=("Segoe UI", 18, "bold"))
        style.configure("HeaderSub.TLabel", background="#102A43", foreground="#CFE4F5",
                        font=("Segoe UI", 10))
        style.configure("Panel.TLabelframe", background="#FFFFFF", borderwidth=1, relief="solid")
        style.configure("Panel.TLabelframe.Label", background="#FFFFFF", foreground="#1F5F99",
                        font=("Segoe UI", 10, "bold"))
        style.configure("Panel.TFrame", background="#FFFFFF")
        style.configure("Body.TLabel", background="#FFFFFF", foreground="#172B3A", font=("Segoe UI", 10))
        style.configure("Muted.TLabel", background="#FFFFFF", foreground="#5B7183", font=("Segoe UI", 9))
        style.configure("Preview.TLabel", background="#FFFFFF", foreground="#5B7183",
                        font=("Segoe UI", 11), padding=20)
        style.configure("Primary.TButton", background="#15803D", foreground="#FFFFFF",
                        font=("Segoe UI", 10, "bold"), padding=(10, 8))
        style.map("Primary.TButton", background=[("active", "#166534"), ("disabled", "#9DB9A7")])
        style.configure("Secondary.TButton", background="#1F5F99", foreground="#FFFFFF",
                        font=("Segoe UI", 10, "bold"), padding=(8, 7))
        style.map("Secondary.TButton", background=[("active", "#174B7A")])
        style.configure("Neutral.TButton", background="#E6EEF5", foreground="#172B3A",
                        font=("Segoe UI", 9), padding=(7, 6))
        style.map("Neutral.TButton", background=[("active", "#D4E2ED")])
        style.configure("Model.TCombobox", fieldbackground="#FFFFFF", background="#1F5F99",
                        foreground="#172B3A", arrowcolor="#FFFFFF", padding=6,
                        font=("Segoe UI", 10, "bold"))
        style.configure("History.Treeview", background="#FFFFFF", fieldbackground="#FFFFFF",
                        foreground="#172B3A", rowheight=28, font=("Segoe UI", 9))
        style.configure("History.Treeview.Heading", background="#1F5F99", foreground="#FFFFFF",
                        font=("Segoe UI", 9, "bold"), relief="flat")
        style.map("History.Treeview", background=[("selected", "#D8EBF8")],
                  foreground=[("selected", "#172B3A")])
        style.configure("Footer.TFrame", background="#102A43")
        style.configure("Footer.TLabel", background="#102A43", foreground="#FFFFFF", font=("Segoe UI", 9))
        style.configure("Green.Horizontal.TProgressbar", troughcolor="#24445E", background="#4ADE80")

    def _susun_layout(self):
        """Menyusun layout tiga panel dengan hierarki visual yang jelas."""
        header = ttk.Frame(self.root, style="Header.TFrame", padding=(22, 15))
        header.pack(fill="x")
        ttk.Label(header, text="DETEKSI MAKANAN INDONESIA", style="HeaderTitle.TLabel").pack(anchor="w")
        ttk.Label(header, text="AI vision untuk klasifikasi citra makanan | EfficientNetB0",
                  style="HeaderSub.TLabel").pack(anchor="w", pady=(2, 0))

        utama = ttk.Frame(self.root, style="App.TFrame", padding=16)
        utama.pack(fill="both", expand=True)
        utama.columnconfigure(0, weight=0, minsize=280)
        utama.columnconfigure(1, weight=3, minsize=370)
        utama.columnconfigure(2, weight=3, minsize=400)
        utama.rowconfigure(0, weight=1)

        kiri = ttk.LabelFrame(utama, text=" MODEL DAN KONTROL ", style="Panel.TLabelframe", padding=14)
        kiri.grid(row=0, column=0, sticky="nsw", padx=(0, 12))
        ttk.Label(kiri, text="Pilih model", style="Body.TLabel").pack(anchor="w")
        self.combo_model = ttk.Combobox(kiri, textvariable=self.var_model, state="readonly",
                                        style="Model.TCombobox", width=28)
        self.combo_model.pack(fill="x", pady=(5, 4))
        self.combo_model.bind("<<ComboboxSelected>>", self._perbarui_detail_model)
        ttk.Label(kiri, textvariable=self.var_model_detail, style="Muted.TLabel",
                  wraplength=250, justify="left").pack(anchor="w", pady=(0, 10))

        baris_model = ttk.Frame(kiri, style="Panel.TFrame")
        baris_model.pack(fill="x", pady=(0, 5))
        ttk.Button(baris_model, text="Refresh", style="Neutral.TButton", command=self._refresh_daftar_model).pack(side="left", expand=True, fill="x", padx=(0, 3))
        ttk.Button(baris_model, text="Browse", style="Neutral.TButton", command=self._browse_model).pack(side="left", expand=True, fill="x", padx=(3, 0))
        self.btn_load = ttk.Button(kiri, text="MUAT MODEL", style="Secondary.TButton", command=self._load_model)
        self.btn_load.pack(fill="x", pady=(4, 5))
        ttk.Button(kiri, text="Info model", style="Neutral.TButton", command=self._tampilkan_info_model).pack(fill="x", pady=(0, 14))

        ttk.Separator(kiri).pack(fill="x", pady=3)
        ttk.Label(kiri, text="Gambar uji", style="Body.TLabel").pack(anchor="w", pady=(12, 3))
        ttk.Button(kiri, text="PILIH GAMBAR", style="Secondary.TButton", command=self._buka_gambar).pack(fill="x", pady=(0, 10))
        baris_topk = ttk.Frame(kiri, style="Panel.TFrame")
        baris_topk.pack(fill="x", pady=(0, 8))
        ttk.Label(baris_topk, text="Top-K prediksi", style="Body.TLabel").pack(side="left")
        ttk.Spinbox(baris_topk, from_=1, to=13, width=5, textvariable=self.var_topk).pack(side="right")
        self.btn_detect = ttk.Button(kiri, text="DETEKSI GAMBAR", style="Primary.TButton", command=self._deteksi, state="disabled")
        self.btn_detect.pack(fill="x", pady=(0, 16))

        ttk.Separator(kiri).pack(fill="x", pady=3)
        ttk.Label(kiri, text="Riwayat", style="Body.TLabel").pack(anchor="w", pady=(12, 3))
        ttk.Button(kiri, text="Ekspor CSV", style="Neutral.TButton", command=self._ekspor_csv).pack(fill="x", pady=(0, 4))
        ttk.Button(kiri, text="Bersihkan riwayat", style="Neutral.TButton", command=self._bersihkan_riwayat).pack(fill="x")

        tengah = ttk.LabelFrame(utama, text=" PREVIEW GAMBAR ", style="Panel.TLabelframe", padding=12)
        tengah.grid(row=0, column=1, sticky="nsew", padx=(0, 12))
        tengah.rowconfigure(0, weight=1)
        tengah.columnconfigure(0, weight=1)
        self.lbl_preview = ttk.Label(tengah, text="Belum ada gambar.\nPilih gambar untuk melihat pratinjau.", style="Preview.TLabel", anchor="center", justify="center")
        self.lbl_preview.grid(row=0, column=0, sticky="nsew")
        self.lbl_nama_file = ttk.Label(tengah, text="", style="Muted.TLabel", anchor="center")
        self.lbl_nama_file.grid(row=1, column=0, sticky="ew", pady=(10, 0))

        kanan = ttk.Frame(utama, style="App.TFrame")
        kanan.grid(row=0, column=2, sticky="nsew")
        kanan.rowconfigure(0, weight=3)
        kanan.rowconfigure(1, weight=2)
        kanan.columnconfigure(0, weight=1)
        frame_hasil = ttk.LabelFrame(kanan, text=" HASIL DETEKSI ", style="Panel.TLabelframe", padding=10)
        frame_hasil.grid(row=0, column=0, sticky="nsew", pady=(0, 12))
        frame_hasil.rowconfigure(0, weight=1)
        frame_hasil.columnconfigure(0, weight=1)
        self.canvas_hasil = Canvas(frame_hasil, bg="#FFFFFF", highlightthickness=0)
        self.canvas_hasil.grid(row=0, column=0, sticky="nsew")

        frame_riwayat = ttk.LabelFrame(kanan, text=" RIWAYAT DETEKSI ", style="Panel.TLabelframe", padding=10)
        frame_riwayat.grid(row=1, column=0, sticky="nsew")
        frame_riwayat.rowconfigure(0, weight=1)
        frame_riwayat.columnconfigure(0, weight=1)
        kolom = ("waktu", "file", "prediksi", "confidence")
        self.tabel_riwayat = ttk.Treeview(frame_riwayat, columns=kolom, show="headings", height=6, style="History.Treeview")
        for nama, lebar in zip(kolom, (72, 150, 125, 92)):
            self.tabel_riwayat.heading(nama, text=nama.capitalize())
            self.tabel_riwayat.column(nama, width=lebar, anchor="w")
        self.tabel_riwayat.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(frame_riwayat, command=self.tabel_riwayat.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.tabel_riwayat.configure(yscrollcommand=scroll.set)

        status = ttk.Frame(self.root, style="Footer.TFrame", padding=(16, 8))
        status.pack(fill="x", side="bottom")
        ttk.Label(status, textvariable=self.var_status, style="Footer.TLabel", anchor="w").pack(side="left")
        self.progress = ttk.Progressbar(status, style="Green.Horizontal.TProgressbar", mode="indeterminate", length=170)
        self.progress.pack(side="right")

    def _refresh_daftar_model(self):
        """Menampilkan nama file model agar pilihan mudah dibaca."""
        daftar = sorted((p for p in MODELS_DIR.iterdir() if p.suffix.lower() in (".keras", ".h5")), key=lambda p: p.name.lower()) if MODELS_DIR.exists() else []
        self._model_paths = {p.name: str(p) for p in daftar}
        labels = list(self._model_paths)
        self.combo_model["values"] = labels
        if labels and self.var_model.get() not in self._model_paths:
            self.combo_model.current(0)
        self._perbarui_detail_model()
        self._set_status(f"Ditemukan {len(labels)} model di folder models/.")

    def _perbarui_detail_model(self, _event=None):
        path = self._model_paths.get(self.var_model.get())
        if path and Path(path).exists():
            file_model = Path(path)
            self.var_model_detail.set(f"{file_model.name} | {file_model.stat().st_size / 1e6:.1f} MB")
        else:
            self.var_model_detail.set("Belum ada model yang dipilih.")

    def _browse_model(self):
        path = filedialog.askopenfilename(title="Pilih file model", filetypes=[("Model Keras", "*.keras *.h5"), ("Semua file", "*.*")], initialdir=MODELS_DIR if MODELS_DIR.exists() else BASE_DIR)
        if path:
            label = Path(path).name
            self._model_paths[label] = path
            self.combo_model["values"] = list(self._model_paths)
            self.var_model.set(label)
            self._perbarui_detail_model()

    def _load_model(self):
        pilihan = self.var_model.get()
        path = self._model_paths.get(pilihan, pilihan)
        if not path:
            messagebox.showwarning("Model belum dipilih", "Pilih file model terlebih dahulu.")
            return
        self.btn_load.config(state="disabled")
        self.btn_detect.config(state="disabled")
        self.progress.start(12)
        self._set_status(f"Memuat model: {Path(path).name} ...")
        threading.Thread(target=self._load_model_worker, args=(path,), daemon=True).start()

    def _gambar_bar_hasil(self, hasil):
        c = self.canvas_hasil
        c.delete("all")
        c.update_idletasks()
        lebar = max(c.winfo_width(), 300)
        margin, tinggi_baris = 18, 54
        c.create_text(margin, margin, anchor="nw", font=("Segoe UI", 12, "bold"), fill="#102A43", text="Top-K prediksi")
        lebar_bar_max = lebar - 2 * margin
        for i, (nama, prob) in enumerate(hasil):
            y = 48 + i * tinggi_baris
            warna = "#15803D" if i == 0 else "#2F80ED"
            c.create_text(margin, y, anchor="nw", font=("Segoe UI", 10, "bold"), fill="#172B3A", text=f"{i + 1}. {nama}   {prob * 100:.2f}%")
            y_bar = y + 24
            c.create_rectangle(margin, y_bar, margin + lebar_bar_max, y_bar + 15, fill="#DCE6EE", outline="")
            c.create_rectangle(margin, y_bar, margin + lebar_bar_max * prob, y_bar + 15, fill=warna, outline="")
    def _load_model_selesai(self, model, class_names, input_size, metadata, path, error):
        """Menerapkan model yang selesai dimuat pada thread UI."""
        self.progress.stop()
        self.btn_load.config(state="normal")
        if error is not None:
            messagebox.showerror("Gagal memuat model", str(error))
            self._set_status("Gagal memuat model.")
            return
        self.model = model
        self.class_names = class_names
        self.input_size = input_size
        self.metadata = metadata
        self.btn_detect.config(state="normal")
        self._set_status(f"Model '{Path(path).name}' dimuat | {len(class_names)} kelas | input {input_size[0]}x{input_size[1]}.")

    def _tampilkan_info_model(self):
        """Menampilkan metadata model dengan format akurasi yang mudah dibaca."""
        if self.model is None:
            messagebox.showinfo("Info Model", "Belum ada model yang dimuat.")
            return
        info = [
            f"Jumlah kelas : {len(self.class_names)}",
            f"Ukuran input : {self.input_size[0]} x {self.input_size[1]}",
            f"Jumlah parameter : {self.model.count_params():,}",
        ]
        if self.metadata:
            akurasi = self.metadata.get("akurasi_test")
            akurasi_teks = f"{float(akurasi) * 100:.2f}%" if isinstance(akurasi, (int, float)) else "-"
            info += [
                f"Akurasi test : {akurasi_teks}",
                f"Tanggal training : {self.metadata.get('tanggal_training', '-')}",
                f"Dataset : {self.metadata.get('dataset_kaggle', '-')}",
            ]
        info.append("\nDaftar kelas:\n" + ", ".join(self.class_names))
        messagebox.showinfo("Info Model", "\n".join(info))
# ------------------------------------------------------------
if __name__ == "__main__":
    root = Tk()
    # Pakai tema 'clam' agar tampilan ttk lebih modern & konsisten antar-OS
    ttk.Style(root).theme_use("clam")
    app = FoodDetectorApp(root)
    root.mainloop()
