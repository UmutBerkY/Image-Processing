import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

# Görüntüyü okuyoruz (gri tonlamalı)
I1 = cv2.imread('lenna.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32)  # Gri tonlamalı okuma ve float32 dönüşümü

# Prewitt filtresini oluşturuyoruz (X ve Y yönü için)
prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])  # Yatay kenar algılama
prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])  # Dikey kenar algılama

# Prewitt filtresi ile görüntü üzerinde işlem yapıyoruz
prewitt_x_result = convolve(I1, prewitt_x)  # X yönü filtresi
prewitt_y_result = convolve(I1, prewitt_y)  # Y yönü filtresi

# Yatay ve dikey kenarların büyüklüğünü hesaplıyoruz
prewitt_magnitude = np.hypot(prewitt_x_result, prewitt_y_result)  # Kenar büyüklüğü
prewitt_magnitude = (prewitt_magnitude / prewitt_magnitude.max()) * 255  # Normalize etme
prewitt_magnitude = np.uint8(prewitt_magnitude)  # Görselleştirme için uint8 dönüşümü

# Sonuçları görselleştiriyoruz
plt.figure(), plt.imshow(prewitt_magnitude, cmap='gray')
plt.title('Prewitt Filtresi Uygulaması')
plt.axis('off')
plt.show()