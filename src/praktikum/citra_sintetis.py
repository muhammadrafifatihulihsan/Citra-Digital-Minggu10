# %% 
# ==================== GENERATE SAMPLE CITRA & SAVE ====================
import cv2
import numpy as np
import os

# Buat folder output bila belum ada
output_dir = 'src\img'
os.makedirs(output_dir, exist_ok=True)

# ============================================================
# CITRA A - Teks dengan noise (titik, goresan, sambungan)
# ============================================================
# Kanvas putih 700 x 200
height_a, width_a = 200, 700
img_teks = np.ones((height_a, width_a), dtype=np.uint8) * 255

# Tulis teks menggunakan font sambung (HERSHEY_SCRIPT_SIMPLEX) agar karakter alami menyambung
teks = "MORFOLOGI"
font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX  # font sambung bawaan OpenCV
font_scale = 1.8
thickness = 3
text_size = cv2.getTextSize(teks, font, font_scale, thickness)[0]
# Posisi tengah
text_x = (width_a - text_size[0]) // 2
text_y = (height_a + text_size[1]) // 2
cv2.putText(img_teks, teks, (text_x, text_y), font, font_scale, 0, thickness, cv2.LINE_AA)

# Tambahkan noise titik (salt & pepper) – 3% dari piksel menjadi hitam
noise_mask = np.random.rand(height_a, width_a) < 0.03
img_teks[noise_mask] = 0

# Tambahkan goresan (garis acak) – 5 garis hitam pendek
for _ in range(5):
    x1, y1 = np.random.randint(0, width_a), np.random.randint(0, height_a)
    x2, y2 = x1 + np.random.randint(-40, 40), y1 + np.random.randint(-40, 40)
    cv2.line(img_teks, (x1, y1), (x2, y2), 0, thickness=2)

# Simpan Citra A
cv2.imwrite(os.path.join(output_dir, 'src/img/citra_teks_noise.png'), img_teks)
print("Citra A (teks + noise) disimpan sebagai 'citra_teks_noise.png'")
print("Ground truth teks yang tertulis: 'MORFOLOGI'\n")

# ============================================================
# CITRA B - Objek overlapping/bersentuhan (lingkaran putih di latar hitam)
# ============================================================
height_b, width_b = 500, 500
img_objek = np.zeros((height_b, width_b), dtype=np.uint8)  # latar hitam

np.random.seed(42)  # agar reproducible
num_circles = 15  # jumlah objek sebenarnya
centers = []
radii = []
for _ in range(num_circles):
    cx = np.random.randint(50, width_b - 50)
    cy = np.random.randint(50, height_b - 50)
    r = np.random.randint(20, 40)
    # Gambar lingkaran putih
    cv2.circle(img_objek, (cx, cy), r, 255, -1)  # -1: filled
    centers.append((cx, cy))
    radii.append(r)

# Opsional: tambahkan sedikit Gaussian noise agar lebih realistis
noise = np.random.normal(0, 10, img_objek.shape).astype(np.int16)
img_objek_noisy = np.clip(img_objek.astype(np.int16) + noise, 0, 255).astype(np.uint8)

# Simpan Citra B (versi tanpa noise dan dengan noise)
cv2.imwrite(os.path.join(output_dir, 'src/img/citra_objek_overlap_clean.png'), img_objek)
cv2.imwrite(os.path.join(output_dir, 'src/img/citra_objek_overlap_noisy.png'), img_objek_noisy)
print(f"Citra B (objek overlapping) disimpan:")
print(f" - versi bersih: 'citra_objek_overlap_clean.png'")
print(f" - versi noise : 'citra_objek_overlap_noisy.png'")
print(f"Ground truth jumlah objek: {num_circles}")

# ============================================================
# Tampilkan kedua citra untuk verifikasi (opsional, jika di notebook)
# ============================================================
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(img_teks, cmap='gray')
axes[0].set_title('Citra A: Teks + Noise')
axes[0].axis('off')
axes[1].imshow(img_objek, cmap='gray')
axes[1].set_title('Citra B: Objek Bersih')
axes[1].axis('off')
axes[2].imshow(img_objek_noisy, cmap='gray')
axes[2].set_title('Citra B: Objek + Noise')
axes[2].axis('off')
plt.tight_layout()
plt.show()