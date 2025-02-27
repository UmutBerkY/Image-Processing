import cv2
import numpy as np
import matplotlib.pyplot as plt

# Görüntüyü okuduk ve double (float) formatına dönüştürdük
I1 = cv2.imread('lenna.jpg', cv2.IMREAD_GRAYSCALE) / 255.0

# 3x3 ve 10x10 filtrelerin oluşturulması
w1 = np.ones((3, 3)) / 9  # Ortalama filtre (3x3)
w2 = np.ones((10, 10)) / 100  # Ortalama filtre (10x10)


# Filtreleri manuel olarak uygulama (kenarları kopyalama ile 'replicate' işlevi benzeri)
def apply_filter(image, kernel):
    h, w = image.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2

    # Sınırları kopyalama
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')

    # Yeni görüntü matrisi oluştur
    filtered_image = np.zeros_like(image)

    # Filtreleme işlemi
    for i in range(h):
        for j in range(w):
            region = padded_image[i:i + kh, j:j + kw]
            filtered_image[i, j] = np.sum(region * kernel)

    return filtered_image


# 3x3 ve 10x10 filtreleri uygula
I2 = apply_filter(I1, w1)  # 3x3 filtre
I3 = apply_filter(I1, w2)  # 10x10 filtre

# Yeni görüntüleri göster
plt.figure(), plt.imshow(I2, cmap='gray'), plt.title('3x3 Filtre Sonucu')
plt.figure(), plt.imshow(I3, cmap='gray'), plt.title('10x10 Filtre Sonucu')
plt.show()
