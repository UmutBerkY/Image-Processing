import cv2
import numpy as np
import matplotlib.pyplot as plt

# Görüntüyü okuyoruz (cv2 sadece okuma işlemi için kullanılıyor)
I1 = cv2.imread('erosion.jpg', cv2.IMREAD_GRAYSCALE)  # Gri tonlamalı okuma

# Kernelin hazırlanması (disk şeklinde 10 birim çapında bir kernel)
kernel = np.ones((21, 21), np.uint8)  # 21x21 boyutunda bir kernel

# Açma (Opening) işlemi: önce erozyon, sonra dilatasyon
# Erozyon
rows, cols = I1.shape
pad_size = kernel.shape[0] // 2
I1_padded = np.pad(I1, pad_size, mode='constant', constant_values=255)
I2_erosion = np.zeros_like(I1)

for i in range(pad_size, rows + pad_size):
    for j in range(pad_size, cols + pad_size):
        region = I1_padded[i - pad_size:i + pad_size + 1, j - pad_size:j + pad_size + 1]
        I2_erosion[i - pad_size, j - pad_size] = np.min(region[kernel == 1])  # Minimum değer

# Dilatasyon
I2_padded = np.pad(I2_erosion, pad_size, mode='constant', constant_values=0)
I2_dilation = np.zeros_like(I2_erosion)

for i in range(pad_size, rows + pad_size):
    for j in range(pad_size, cols + pad_size):
        region = I2_padded[i - pad_size:i + pad_size + 1, j - pad_size:j + pad_size + 1]
        I2_dilation[i - pad_size, j - pad_size] = np.max(region[kernel == 1])  # Maksimum değer

# Kapama (Closing) işlemi: önce dilatasyon, sonra erozyon
# Dilatasyon
I1_padded = np.pad(I1, pad_size, mode='constant', constant_values=0)
I3_dilation = np.zeros_like(I1)

for i in range(pad_size, rows + pad_size):
    for j in range(pad_size, cols + pad_size):
        region = I1_padded[i - pad_size:i + pad_size + 1, j - pad_size:j + pad_size + 1]
        I3_dilation[i - pad_size, j - pad_size] = np.max(region[kernel == 1])  # Maksimum değer

# Erozyon
I3_padded = np.pad(I3_dilation, pad_size, mode='constant', constant_values=255)
I3_erosion = np.zeros_like(I3_dilation)

for i in range(pad_size, rows + pad_size):
    for j in range(pad_size, cols + pad_size):
        region = I3_padded[i - pad_size:i + pad_size + 1, j - pad_size:j + pad_size + 1]
        I3_erosion[i - pad_size, j - pad_size] = np.min(region[kernel == 1])  # Minimum değer

# Sonuçları matplotlib ile gösteriyoruz
plt.figure(), plt.imshow(I1, cmap='gray')
plt.title('Orijinal Görüntü')

plt.figure(), plt.imshow(I2_dilation, cmap='gray')
plt.title('Açma (Opening) İşlemi')

plt.figure(), plt.imshow(I3_erosion, cmap='gray')
plt.title('Kapama (Closing) İşlemi')

plt.show()