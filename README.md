# Pengolahan Citra Digital - Praktikum & Assignment Minggu 10
**Topik: Morfologi Citra & Segmentasi Dasar (Image Morphology & Basic Segmentation)**

Repository ini berisi kumpulan kode program untuk mata kuliah Pengolahan Citra Digital (PCD) pada Minggu ke-10. Fokus utama pada minggu ini adalah pendalaman teknik morfologi citra untuk ekstraksi fitur, pembersihan noise, dan segmentasi objek. Berbagai metode yang dikembangkan meliputi operasi dasar (Erosi, Dilasi), operasi majemuk (Opening, Closing, Morphological Gradient, Top-hat), hingga aplikasi nyata pada preprocessing OCR dan perhitungan jumlah objek menggunakan algoritma Watershed.

## 👤 Informasi Mahasiswa

*   **Nama:** Muhammad Rafi Fatihul Ihsan
*   **NIM:** 24343016
*   **Sesi/Kelas:** 202523430039

## 📂 Struktur Project

```text
program/
├── src/
│   ├── assignment/
│   │   ├── assignment.ipynb    # Assignment Utama: Pipeline Morfologi untuk OCR & Counting
│   │   └── assignment.py       # Script Python mandiri untuk Pipeline Morfologi
│   ├── praktikum/
│   │   ├── citra_sintetis.py   # Generator Citra Uji (Teks Noise & Objek Overlap)
│   │   ├── praktikum1.ipynb    # Eksperimen Operasi Morfologi Dasar & Majemuk
│   │   └── praktikum2.ipynb    # Implementasi Segmentasi Watershed & Analisis SE
│   └── img/                    # Dataset citra sintetis dan hasil generasi
├── requirements.txt            # Daftar dependensi Python
└── README.md                   # Dokumentasi proyek
```

## 🚀 Fitur & Modul Praktikum

**1. Generator Citra Sintetis (`citra_sintetis.py`)**
Modul ini digunakan untuk menciptakan dataset uji secara otomatis. Menghasilkan dua jenis citra utama:
*   **Citra Teks Ber-noise:** Teks "MORFOLOGI" dengan gangguan *salt & pepper* dan goresan garis untuk menguji efektivitas pembersihan noise.
*   **Citra Objek Overlapping:** Kumpulan lingkaran yang saling bersentuhan untuk menguji kemampuan segmentasi pemisah objek.

**2. Eksperimen Morfologi Dasar & Majemuk (`praktikum1.ipynb`)**
Mendemonstrasikan perilaku berbagai operator morfologi terhadap citra biner dan grayscale:
*   **Dasar:** Erosi (pengecilan objek) dan Dilasi (penebalan objek).
*   **Majemuk:** *Opening* untuk menghilangkan noise halus dan *Closing* untuk menutup lubang/keretakan pada struktur objek.
*   **Ekstraksi:** Menggunakan *Morphological Gradient* untuk mendapatkan *boundary* (tepi) objek secara presisi.

**3. Analisis Structuring Element (SE) & Watershed (`praktikum2.ipynb`)**
Fokus pada pemilihan kernel (SE) dan teknik segmentasi lanjut:
*   Perbandingan bentuk SE (*Square, Cross, Ellipse*) dan pengaruh ukuran kernel terhadap distorsi bentuk asli.
*   Implementasi **Watershed Segmentation** dengan bantuan *Distance Transform* untuk memisahkan objek yang menempel atau *overlapping*.

## 📝 Assignment Utama: Pipeline Morfologi untuk OCR & Counting
Tugas utama minggu ini adalah membangun pipeline pengolahan citra sekuensial yang menggabungkan berbagai teknik morfologi untuk menyelesaikan dua problem spesifik:

1.  **Preprocessing OCR (Optical Character Recognition):**
    Menggunakan kombinasi *Opening* dan *Closing* untuk membersihkan naskah dokumen yang rusak sehingga meningkatkan akurasi pembacaan teks oleh engine Tesseract.
2.  **Object Counting (Perhitungan Objek):**
    Mengimplementasikan pipeline otomatis untuk menghitung jumlah koin atau sel pada citra mikroskopis. Menggunakan *Distance Transform* untuk menemukan pusat massa objek dan *Watershed* untuk delineasi batas antar objek yang bersentuhan.
3.  **Analisis Trade-off:**
    Mengevaluasi waktu komputasi dari setiap operasi dan menganalisis dampak pemilihan parameter (seperti iterasi dan ukuran kernel) terhadap kualitas restorasi citra.

## 🛠️ Cara Menjalankan Program

### 1. Instalasi Dependensi
Pastikan Python telah terinstal, lalu instal seluruh pustaka yang dibutuhkan menggunakan `pip`:
```bash
pip install -r requirements.txt
```
*Catatan: Modul OCR membutuhkan engine Tesseract yang terinstal di sistem operasi Anda (Linux: `sudo apt install tesseract-ocr`, Windows: install `.exe` dari UB Mannheim).*

### 2. Menjalankan Skrip Praktikum
Langkah awal adalah menghasilkan citra uji, kemudian menjalankan analisis:
```bash
python src/praktikum/citra_sintetis.py
```
Anda kemudian dapat membuka file `.ipynb` di folder `src/praktikum/` menggunakan VS Code atau Jupyter Notebook.

### 3. Membuka Assignment
Jalankan notebook assignment untuk melihat visualisasi langkah-demi-langkah:
```bash
jupyter notebook src/assignment/assignment.ipynb
```
Atau jalankan versi script langsung:
```bash
python src/assignment/assignment.py
```
