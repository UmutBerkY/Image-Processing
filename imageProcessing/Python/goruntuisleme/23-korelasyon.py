import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Görüntüleri yükle
image = cv2.imread('matkapuclar.png', cv2.IMREAD_GRAYSCALE)
template = cv2.imread('saglamuc.png', cv2.IMREAD_GRAYSCALE)

# Şablon boyutları
th, tw = template.shape

# Korelasyon hesaplama (manuel NCC)
result = np.zeros((image.shape[0] - th + 1, image.shape[1] - tw + 1))
for i in range(result.shape[0]):
    for j in range(result.shape[1]):
        region = image[i:i + th, j:j + tw]
        numerator = np.sum((region - region.mean()) * (template - template.mean()))
        denominator = np.sqrt(np.sum((region - region.mean())**2) * np.sum((template - template.mean())**2))
        result[i, j] = numerator / denominator if denominator != 0 else 0

# En yüksek korelasyon değerinin bulunduğu konumu bul ve işaretle
top_left = np.unravel_index(np.argmax(result), result.shape)
bottom_right = (top_left[1] + tw, top_left[0] + th)

# İşaretli görüntü
marked_image = np.copy(image)
marked_image[top_left[0]:bottom_right[1], top_left[1]] = 255  # Sol kenar
marked_image[top_left[0]:bottom_right[1], bottom_right[0] - 1] = 255  # Sağ kenar
marked_image[top_left[0], top_left[1]:bottom_right[0]] = 255  # Üst kenar
marked_image[bottom_right[1] - 1, top_left[1]:bottom_right[0]] = 255  # Alt kenar

# Görselleştirme
plt.figure(figsize=(12, 6))

# Korelasyon imgesi
plt.subplot(1, 3, 1)
plt.imshow(result, cmap='gray')
plt.title('Korelasyon İmgesi')

# İşaretli görüntü
plt.subplot(1, 3, 2)
plt.imshow(marked_image, cmap='gray')
plt.title('Bulunan Bölge')

# 3D yüzey grafiği
ax = plt.subplot(1, 3, 3, projection='3d')
X, Y = np.meshgrid(np.arange(result.shape[1]), np.arange(result.shape[0]))
ax.plot_surface(X, Y, result, cmap='gray', edgecolor='none')
ax.set_title('Korelasyon Yüzeyi')
ax.set_xlabel('X')
ax.set_ylabel('Y')

plt.tight_layout()
plt.show()