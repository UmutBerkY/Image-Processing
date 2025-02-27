import cv2
import numpy as np
import matplotlib.pyplot as plt

# Görüntüyü okuyoruz
I1 = cv2.imread('lena.bmp')  # Renkli okuma

# Yatay çevrilmiş görüntü (soldan sağa)
I2 = np.fliplr(I1)  # Yatay eksende çevirme

# Dikey çevrilmiş görüntü (yukarıdan aşağıya)
I3 = np.flipud(I1)  # Dikey eksende çevirme

# Hem yatay hem de dikey çevrilmiş görüntü
I4 = np.flipud(np.fliplr(I1))  # Hem yatay hem de dikey eksende çevirme

# BGR -> RGB dönüşümü NumPy kullanılarak yapılır
def bgr_to_rgb(image):
    return image[:, :, ::-1]  # Son eksende ters çevirme (BGR -> RGB)

# Sonuçları görselleştiriyoruz
plt.figure(), plt.imshow(bgr_to_rgb(I2))
plt.title('Yatay Çevirme')
plt.figure(), plt.imshow(bgr_to_rgb(I3))
plt.title('Dikey Çevirme')
plt.figure(), plt.imshow(bgr_to_rgb(I4))
plt.title('Yatay ve Dikey Çevirme')
plt.show()
