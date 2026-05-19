# Pertemuan 8 - Praktikum Kecerdasan Buatan (JST & Keras)
# Muhammad Rezqy Robiansyah
# H1D024129

Repositori ini berisi file-file dan implementasi untuk Praktikum Kecerdasan Buatan (KB) Pertemuan 8. Fokus utamanya adalah implementasi Jaringan Saraf Tiruan (Artificial Neural Network) menggunakan library Keras dan TensorFlow untuk melakukan klasifikasi pada dataset Iris.

Berikut adalah penjelasan lengkap mengenai program dan file yang terdapat dalam folder `pertemuan8`:

## 1. `praktikum7_keras.py`
File ini adalah program utama (Python script) yang mengimplementasikan Jaringan Saraf Tiruan (JST) menggunakan Keras dari TensorFlow untuk mengklasifikasikan spesies bunga Iris. 
Alur kerja program ini meliputi:
- **Loading Dataset**: Membaca dataset Iris dari lokal (mencari file `iris.data` atau `bezdekIris.data` di dalam folder `iris`).
- **Preprocessing Data**: Memisahkan fitur (ukuran sepal & petal) dan label target (spesies). Mengubah label string menjadi numerik (0, 1, 2) menggunakan `LabelEncoder` dari scikit-learn, lalu membagi data menjadi data latih (80%) dan data uji (20%).
- **Membangun Model (Arsitektur JST)**: Membuat model *Sequential* dengan:
  - 1 Layer input (menyesuaikan jumlah fitur).
  - 3 Hidden layer (`Dense`) dengan neuron masing-masing 1000, 500, dan 300 yang menggunakan fungsi aktivasi `relu`.
  - 1 Layer output (`Dense`) dengan 3 neuron (sesuai jumlah spesies) yang menggunakan fungsi aktivasi `softmax` untuk menghasilkan probabilitas tiap kelas.
- **Training (Pelatihan)**: Mengkompilasi model menggunakan optimizer `adam` dan loss function `sparse_categorical_crossentropy`, kemudian melatihnya selama 50 epoch.
- **Evaluasi & Visualisasi**: Menghitung *Loss* dan *Accuracy* dari data uji, lalu memvisualisasikan grafik *Training History* (menggunakan matplotlib) dan *Confusion Matrix* (menggunakan seaborn).
- **Prediksi Interaktif**: Menyediakan fitur di mana user dapat menginput nilai *sepal length*, *sepal width*, *petal length*, dan *petal width* secara manual di terminal untuk diprediksi spesiesnya oleh model.

## 2. Folder `iris`
Folder ini berisi dataset klasik "Iris Plants Database" yang digunakan untuk melatih model pada program di atas. Terdiri dari beberapa file:

### a. `iris.data`
Ini adalah dataset asli Iris yang berisi 150 baris data (instances). Setiap baris memiliki 5 kolom yang dipisahkan oleh koma:
1. *Sepal length* (cm)
2. *Sepal width* (cm)
3. *Petal length* (cm)
4. *Petal width* (cm)
5. *Class* / Spesies (Iris-setosa, Iris-versicolor, atau Iris-virginica).

### b. `bezdekIris.data`
File ini merupakan versi dataset Iris yang telah dikoreksi. Seperti yang dijelaskan di file `iris.names`, terdapat dua kesalahan ketik kecil pada data yang diterbitkan oleh R.A. Fisher (pada baris ke-35 dan ke-38). `bezdekIris.data` digunakan jika ingin menggunakan versi dataset dengan nilai atribut yang sudah diperbaiki.

### c. `iris.names`
File ini adalah dokumentasi atau *metadata* resmi dari dataset Iris. Berisi informasi lengkap mengenai:
- Pencipta dataset (R.A. Fisher) dan sumbernya.
- Riwayat publikasi dan penggunaan masa lalu dari dataset ini.
- Informasi mendetail mengenai atribut (fitur), jumlah kelas, dan distribusi kelas.
- Statistik ringkas seperti nilai minimum, maksimum, rata-rata, dan standar deviasi dari setiap fitur.
- Catatan mengenai perbaikan data pada sampel ke-35 dan ke-38 (yang menjadi alasan adanya file `bezdekIris.data`).

### d. `Index`
File teks sederhana yang berfungsi sebagai indeks direktori dari arsip UCI Machine Learning Repository (tempat dataset ini berasal). File ini mencantumkan daftar file (`Index`, `iris.data`, `iris.names`) beserta tanggal modifikasi dan ukuran filenya.