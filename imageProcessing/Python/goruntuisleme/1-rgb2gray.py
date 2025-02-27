import cv2
import numpy as np

# Renkli resmi yükle
resim = cv2.imread('lena.bmp')

# Gri tonlamayı manuel olarak yap
# Resmin boyutlarını al
h, w, c = resim.shape

# Gri tonlama resmi oluştur
# Her piksel için gri değer: 0.299 * R + 0.587 * G + 0.114 * B
gri_resim = np.zeros((h, w), dtype=np.uint8)

for i in range(h):
    for j in range(w):
        r, g, b = resim[i, j]  # Piksel değerlerini al
        gri_deger = int(0.299 * r + 0.587 * g + 0.114 * b)  # Gri değer hesapla
        gri_resim[i, j] = gri_deger  # Gri değeri ata

# Sonucu göster
cv2.imshow('Renkli Resim', resim)
cv2.imshow('Gri Tonlama Resim', gri_resim)
cv2.waitKey(0)
cv2.destroyAllWindows()
