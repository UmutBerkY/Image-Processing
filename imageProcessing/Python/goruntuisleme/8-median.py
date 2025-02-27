import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import random_noise

# Görüntüyü okuduk.
I1 = cv2.imread('lenna.jpg', cv2.IMREAD_GRAYSCALE)
# Siyah beyaz noktaları (salt & pepper) ekledik.
I2 = random_noise(I1, mode='s&p', amount=0.02)  # %2 oranında 'salt & pepper' gürültüsü ekleniyor.
I2 = np.array(255 * I2, dtype=np.uint8)  # Gürültüden sonra görüntüyü yeniden 0-255 aralığına getiriyoruz.
# Median filtreyi manuel olarak uygulama
def median_filter_manual(image, size):
    h, w = image.shape
    pad = size // 2
    # Sınırları kopyalama
    padded_image = np.pad(image, pad, mode='edge')
    # Yeni görüntü matrisi oluştur
    filtered_image = np.zeros_like(image)
    # Median filtreleme işlemi
    for i in range(h):
        for j in range(w):
            region = padded_image[i:i+size, j:j+size]
            filtered_image[i, j] = np.median(region)
    return filtered_image
# Median filtre uygula
I3 = median_filter_manual(I2, size=3)  # 3x3 boyutunda median filtre
# Sonuçları gösterdik
plt.figure(), plt.imshow(I2, cmap='gray')
plt.title('Gürültü Eklenmiş Görüntü')
plt.figure(), plt.imshow(I3, cmap='gray')
plt.title('Median Filtre Uygulanmış Görüntü')

plt.show()
