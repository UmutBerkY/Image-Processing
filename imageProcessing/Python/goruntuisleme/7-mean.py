import cv2
import numpy as np
import matplotlib.pyplot as plt

# Görüntüyü okuduk ve double tipine dönüştürdük.
I1 = cv2.imread('lenna.jpg', cv2.IMREAD_GRAYSCALE) / 255.0  # Gri tonlama

# Ortalama (average) filtreyi manuel olarak oluşturma (10x10 boyutunda)
kernel_size = (10, 10)
average_kernel = np.ones(kernel_size) / (kernel_size[0] * kernel_size[1])  # 10x10 boyutunda ortalama filtre

# Filtreyi manuel olarak uygulama (kenarları kopyalama ile 'replicate' işlevi benzeri)
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
            region = padded_image[i:i+kh, j:j+kw]
            filtered_image[i, j] = np.sum(region * kernel)

    return filtered_image

# Ortalama filtreyi uygula
I2 = apply_filter(I1, average_kernel)

# Sonucu göster
plt.figure(), plt.imshow(I2, cmap='gray')
plt.title('Ortalama Filtre Uygulaması')
plt.show()
