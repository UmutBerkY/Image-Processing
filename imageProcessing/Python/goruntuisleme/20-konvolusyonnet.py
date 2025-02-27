import cv2
import numpy as np
import matplotlib.pyplot as plt

# Görüntüyü okuyoruz (cv2 sadece okuma işlemi için kullanılıyor)
I1 = cv2.imread('lenna.jpg', cv2.IMREAD_GRAYSCALE)  # Gri tonlamalı okuma

# Netleştirme (sharpening) çekirdeğini oluşturuyoruz
sharpen_kernel = np.array([[ 0, -1,  0],
                           [-1,  5, -1],
                           [ 0, -1,  0]], dtype=np.float32)

# Konvolüsyon işlemi için her bir piksel üzerinde kernel'i kaydırıyoruz
pad_size = 1
I1_padded = np.pad(I1, pad_size, mode='constant', constant_values=0)
sharpened_image = np.zeros_like(I1, dtype=np.float32)

# Uygulama için her bir piksel üzerinde işlem yapıyoruz
for i in range(pad_size, I1.shape[0] + pad_size):
    for j in range(pad_size, I1.shape[1] + pad_size):
        region = I1_padded[i-1:i+2, j-1:j+2]
        sharpened_image[i-1, j-1] = np.sum(region * sharpen_kernel)

# Görüntüdeki değerlerin sınırlarını kontrol ediyoruz ve normalize ediyoruz
sharpened_image = np.clip(sharpened_image, 0, 255).astype(np.uint8)

# Sonuçları görselleştiriyoruz
plt.figure(), plt.imshow(I1, cmap='gray')
plt.title('Orijinal Görüntü')

plt.figure(), plt.imshow(sharpened_image, cmap='gray')
plt.title('Netleştirilmiş Görüntü (Konvolüsyon)')

plt.show()
