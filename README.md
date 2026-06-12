# Klasifikasi Gambar Rock-Paper-Scissors Menggunakan CNN (Convolutional Neural Network)
# Muhammad Rezqy Robiansyah - H1D024129
Repository ini berisi kode program Python untuk melatih model pembelajaran mendalam (*deep learning*) guna mengklasifikasikan gambar Tangan Kertas, Batu, dan Gunting (*Rock-Paper-Scissors*). Proyek ini memanfaatkan pustaka TensorFlow dan Keras untuk membuat arsitektur Convolutional Neural Network (CNN).

---

## 📁 Struktur Direktori

Berikut adalah struktur folder dan berkas pada direktori [pertemuan8](file:///c:/prakkb/pertemuan8/):

```text
pertemuan8/
├── praktikum8_cnn_rps.py     # Script utama untuk pelatihan dan evaluasi model CNN
├── requirements.txt          # Daftar pustaka (dependencies) yang dibutuhkan
├── rockpaperscissors/        # Folder dataset utama
│   ├── paper/                # Gambar dataset kategori kertas (paper)
│   ├── rock/                 # Gambar dataset kategori batu (rock)
│   └── scissors/             # Gambar dataset kategori gunting (scissors)
└── README.md                 # Dokumentasi penjelasan program (berkas ini)
```

---

## ⚙️ Prasyarat & Cara Instalasi

Sebelum menjalankan program, pastikan Anda telah menginstal Python (disarankan versi 3.8 - 3.11) dan menginstal pustaka yang tertera di [requirements.txt](file:///c:/prakkb/pertemuan8/requirements.txt):

1. Buka terminal atau Command Prompt pada direktori `pertemuan8`.
2. Jalankan perintah berikut untuk menginstal dependensi:
   ```bash
   pip install -r requirements.txt
   ```

Pustaka utama yang digunakan adalah:
* **TensorFlow** (untuk pemodelan dan pelatihan CNN)
* **NumPy** (untuk komputasi matriks dan manipulasi array)
* **Pandas** (untuk representasi tabel matriks kebingungan / *confusion matrix*)

---

## 🚀 Cara Menjalankan Program

Untuk melatih model dan melihat metrik hasil prediksi, jalankan skrip Python [praktikum8_cnn_rps.py](file:///c:/prakkb/pertemuan8/praktikum8_cnn_rps.py) dengan perintah:

```bash
python praktikum8_cnn_rps.py
```

> [!NOTE]
> Program secara otomatis mendeteksi folder dataset [rockpaperscissors](file:///c:/prakkb/pertemuan8/rockpaperscissors). Jika folder tersebut tidak ada, program akan memunculkan pesan error `FileNotFoundError`.

---

## 🧠 Penjelasan Alur dan Logika Kode

Alur eksekusi di dalam berkas [praktikum8_cnn_rps.py](file:///c:/prakkb/pertemuan8/praktikum8_cnn_rps.py) dapat dibagi menjadi beberapa tahap berikut:

### 1. Augmentasi & Pembagian Dataset (`ImageDataGenerator`)
Untuk memproses dataset gambar dengan efisien tanpa memuat semua gambar langsung ke RAM, digunakan pustaka `ImageDataGenerator` dari Keras:
* **Rescaling**: Nilai piksel gambar dinormalisasi dari rentang `[0, 255]` menjadi `[0.0, 1.0]` dengan mengalikan `1.0 / 255.0`.
* **Validation Split**: Dataset dibagi menjadi data pelatihan sebesar **80%** dan data validasi sebesar **20%** secara acak dengan parameter `validation_split=0.2`.
* **Generator Pembaca Folder**:
  - `train_generator`: Mengambil subset `"training"` dengan target ukuran gambar `150x150` piksel, batch size `32`, dan tipe kelas `categorical`.
  - `validation_generator`: Mengambil subset `"validation"` dengan setelan serupa, namun parameter `shuffle=False` agar hasil prediksi di akhir dapat dipetakan secara berurutan dengan label aslinya.

### 2. Arsitektur Model CNN
Model CNN dibangun secara berurutan (*Sequential*) dengan susunan lapisan (*layer*) sebagai berikut:

| Nama Layer | Konfigurasi / Parameter | Fungsi / Deskripsi |
| :--- | :--- | :--- |
| **Input Layer** | Shape: `(150, 150, 3)` | Menentukan dimensi input gambar yaitu 150x150 piksel dengan 3 saluran warna (RGB). |
| **Conv2D (1)** | 32 filter, ukuran kernel `3x3`, fungsi aktivasi `ReLU` | Mengekstraksi fitur sederhana (seperti tepi dan sudut) dari gambar input. |
| **MaxPooling2D (1)**| Pool size `2x2` | Mengurangi resolusi spasial gambar sebesar setengahnya guna mengurangi komputasi dan menghindari *overfitting*. |
| **Conv2D (2)** | 64 filter, ukuran kernel `3x3`, fungsi aktivasi `ReLU` | Mengekstraksi fitur gambar yang lebih kompleks. |
| **MaxPooling2D (2)**| Pool size `2x2` | Mengurangi dimensi spasial gambar hasil ekstraksi lapisan kedua. |
| **Conv2D (3)** | 128 filter, ukuran kernel `3x3`, fungsi aktivasi `ReLU` | Mengekstraksi pola tingkat tinggi dari objek tangan (batu, kertas, atau gunting). |
| **MaxPooling2D (3)**| Pool size `2x2` | Reduksi dimensi akhir sebelum dilakukan perataan (*flattening*). |
| **Flatten** | - | Mengubah matriks fitur 2D multi-channel menjadi vektor 1D satu baris. |
| **Dense (Hidden)** | 512 unit/neuron, fungsi aktivasi `ReLU` | Lapisan terhubung penuh (*fully connected*) untuk mempelajari kombinasi fitur non-linear tingkat tinggi. |
| **Dense (Output)** | 3 unit/neuron, fungsi aktivasi `Softmax` | Menghasilkan probabilitas distribusi untuk 3 kelas output (kertas, batu, gunting). |

### 3. Kompilasi & Pelatihan Model
* **Loss Function**: Menggunakan `categorical_crossentropy` karena masalah klasifikasi multi-kelas dengan format output berupa *one-hot encoding*.
* **Optimizer**: `adam` (Adaptive Moment Estimation) untuk pembaruan bobot model secara adaptif dan efisien.
* **Metrics**: `['accuracy']` untuk mengukur akurasi model selama proses pelatihan berlangsung.
* **Epochs**: Pelatihan dijalankan sebanyak **10 epoch**.

### 4. Evaluasi & Analisis Prediksi
Setelah pelatihan selesai, skrip melakukan evaluasi mendalam pada data validasi:
* **Evaluasi Loss & Akurasi Validasi**: Menggunakan `model.evaluate` untuk mengukur performa model secara langsung pada data validasi.
* **Prediksi Kelas**: Menggunakan `model.predict` untuk mendapatkan probabilitas untuk setiap gambar validasi, kemudian kelas dengan probabilitas tertinggi dipilih menggunakan `predictions.argmax(axis=1)`.
* **Confusion Matrix (Matriks Kebingungan)**: Membuat matriks kebingungan menggunakan `tf.math.confusion_matrix` dan menampilkannya sebagai objek `pandas.DataFrame`. Matriks ini membantu melihat detail performa klasifikasi untuk setiap kelas (misal, seberapa sering kertas salah diprediksi sebagai gunting).
* **Sampel Hasil Prediksi**: Menampilkan 10 sampel gambar pertama dari data validasi, lengkap dengan label asli (*true label*), label prediksi (*predicted label*), dan persentase tingkat kepercayaan (*confidence score*).
