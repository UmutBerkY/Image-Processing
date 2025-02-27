import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

# Görüntüyü okuyoruz (gri tonlamalı)
I1 = cv2.imread('lenna.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32)  # Gri tonlamalı okuma ve float32 dönüşümü

# Sobel filtresini oluşturuyoruz (X ve Y yönü için)
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # Yatay kenar algılama
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])  # Dikey kenar algılama

# Sobel filtresi ile görüntü üzerinde işlem yapıyoruz
sobel_x_result = convolve(I1, sobel_x)  # X yönü filtresi
sobel_y_result = convolve(I1, sobel_y)  # Y yönü filtresi

# Yatay ve dikey kenarların büyüklüğünü hesaplıyoruz
sobel_magnitude = np.hypot(sobel_x_result, sobel_y_result)  # Kenar büyüklüğü
sobel_magnitude = (sobel_magnitude / sobel_magnitude.max()) * 255  # Normalize etme
sobel_magnitude = np.uint8(sobel_magnitude)  # Görselleştirme için uint8 dönüşümü

# Sonuçları görselleştiriyoruz
plt.figure(), plt.imshow(sobel_magnitude, cmap='gray')
plt.title('Sobel Filtresi Uygulaması')
plt.axis('off')
plt.show()