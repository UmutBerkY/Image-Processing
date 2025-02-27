import cv2
import numpy as np
import matplotlib.pyplot as plt

I1 = cv2.imread('lenna.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float64) / 255.0
# Filtrelerin oluşturulması
laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])  # Laplacian kernel
w8 = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])  # 3x3 özelleştirilmiş filtre
# Filtrelerin manuel olarak uygulanması (kenarları kopyalama ile 'replicate' işlevi benzeri)
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
# Filtrelerin uygulanması
g = apply_filter(I1, laplacian_kernel)
g4 = I1 - g
g8 = I1 - apply_filter(I1, w8)
# Yeni görüntüleri gösterdik
plt.figure(), plt.imshow(g, cmap='gray')
plt.title('Laplacian Filtre Uygulaması')
plt.figure(), plt.imshow(g4, cmap='gray')
plt.title('I1 - Laplacian Filtre')
plt.figure(), plt.imshow(g8, cmap='gray')
plt.title('I1 - Özelleştirilmiş 3x3 Filtre')
plt.show()
