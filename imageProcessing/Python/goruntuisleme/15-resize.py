import cv2
import numpy as np
import matplotlib.pyplot as plt

# Görüntüyü okuyoruz (cv2 sadece okuma işlemi için kullanılıyor)
I1 = cv2.imread('lena.bmp')  # Renkli okuma
I1 = I1[:, :, ::-1]  # BGR -> RGB dönüşümü

# Görüntüyü yeniden boyutlandırıyoruz (numpy ile)
height, width, channels = I1.shape
new_height, new_width = 100, 150  # İstenen boyutlar
I2 = np.zeros((new_height, new_width, channels), dtype=I1.dtype)  # Boş bir array oluşturma

# Her pikselin yeni pozisyonunu hesaplayarak görüntüyü yeniden boyutlandırıyoruz
scale_y, scale_x = height / new_height, width / new_width
for y in range(new_height):
    for x in range(new_width):
        src_y = int(y * scale_y)
        src_x = int(x * scale_x)
        I2[y, x] = I1[src_y, src_x]

# İlk ve ikinci görüntüleri matplotlib ile gösteriyoruz
plt.figure()
plt.imshow(I1)
plt.title('Orijinal Görüntü')
plt.axis('off')

plt.figure()
plt.imshow(I2)
plt.title('Yeniden Boyutlandırılmış Görüntü')
plt.axis('off')

plt.show()