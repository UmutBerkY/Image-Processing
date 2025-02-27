import cv2
import numpy as np
import matplotlib.pyplot as plt

# Görüntüyü yükle
griresim = cv2.imread('lenna.jpg', cv2.IMREAD_GRAYSCALE)

# Sabit eşiklemeyi manuel olarak yap
# Yeni bir ikili resim matrisi oluştur
h, w = griresim.shape
ikili_resim = np.zeros((h, w), dtype=np.uint8)

# Eşikleme değeri
esik = 128

# Her piksel için eşikleme uygula
for i in range(h):
    for j in range(w):
        if griresim[i, j] >= esik:
            ikili_resim[i, j] = 255
        else:
            ikili_resim[i, j] = 0

# Sonucu göster
plt.imshow(ikili_resim, cmap='gray')
plt.title('Sabit Eşikleme Sonucu')
plt.show()
