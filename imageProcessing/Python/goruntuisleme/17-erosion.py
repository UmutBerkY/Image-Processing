import cv2
import numpy as np
import matplotlib.pyplot as plt

# Görüntüyü okuyoruz (cv2 sadece okuma işlemi için kullanılıyor)
I1 = cv2.imread('erosion.jpg', cv2.IMREAD_GRAYSCALE)  # Gri tonlamalı okuma

# Kernelin hazırlanması (3x3 boyutunda)
kernel = np.ones((3, 3), np.uint8)  # 3x3 boyutunda bir kernel (çekirdek)

# Erozyon işlemi için numpy ile bir fonksiyon oluşturuyoruz
rows, cols = I1.shape
pad_size = kernel.shape[0] // 2
I_padded = np.pad(I1, pad_size, mode='constant', constant_values=255)  # Padding
I2 = np.zeros_like(I1)  # Çıktı görüntüsü

# Erozyon işlemi
for i in range(pad_size, rows + pad_size):
    for j in range(pad_size, cols + pad_size):
        region = I_padded[i - pad_size:i + pad_size + 1, j - pad_size:j + pad_size + 1]
        if np.all(region[kernel == 1] == 255):  # Kernelin tam örtüşmesi gerekiyor
            I2[i - pad_size, j - pad_size] = 255
        else:
            I2[i - pad_size, j - pad_size] = 0

# Görüntüleri matplotlib ile gösteriyoruz
plt.figure(), plt.imshow(I1, cmap='gray')
plt.title('Orijinal Görüntü')

plt.figure(), plt.imshow(I2, cmap='gray')
plt.title('Erozyon İşlemi')

plt.show()