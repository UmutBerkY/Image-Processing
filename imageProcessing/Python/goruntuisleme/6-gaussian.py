import cv2
import numpy as np
import matplotlib.pyplot as plt

I1 = cv2.imread('lenna.jpg', cv2.IMREAD_GRAYSCALE) / 255.0  # Gri tonlama

# Gaussian filtreyi manuel olarak oluşturma (10x10 boyutunda, sigma=2)
def gaussian_kernel(size, sigma):
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))

    return kernel / np.sum(kernel)

kernel_size = 10
sigma = 2
gaussian_kernel = gaussian_kernel(kernel_size, sigma)

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

# Gaussian filtreyi uygula
I2 = apply_filter(I1, gaussian_kernel)

# Sonuçları göster
plt.figure(), plt.imshow(I2, cmap='gray')
plt.title('Gaussian Filtre Uygulaması')
plt.show()
