import numpy as np
import matplotlib.pyplot as plt
import cv2

# Görüntüyü okuyoruz
I1 = cv2.imread('lenna.jpg', cv2.IMREAD_GRAYSCALE)

# Sobel operatörünü manuel olarak uyguluyoruz

# Sobel X yönü (Yatay kenarları algılamak için)
sobel_x_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
sobel_x = np.zeros_like(I1, dtype=np.float32)

# Sobel Y yönü (Dikey kenarları algılamak için)
sobel_y_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
sobel_y = np.zeros_like(I1, dtype=np.float32)

# Uygulama için kernel'i her bir piksel üzerinde kaydırıyoruz
pad_size = 1
I1_padded = np.pad(I1, pad_size, mode='constant', constant_values=0)

for i in range(pad_size, I1.shape[0] + pad_size):
    for j in range(pad_size, I1.shape[1] + pad_size):
        region = I1_padded[i-1:i+2, j-1:j+2]
        sobel_x[i-1, j-1] = np.sum(region * sobel_x_kernel)
        sobel_y[i-1, j-1] = np.sum(region * sobel_y_kernel)

# Kenarları birleştiriyoruz (magnitude hesaplamak)
edges = np.hypot(sobel_x, sobel_y)

# Kenarları normalize ediyoruz
edges = np.uint8(np.absolute(edges))

# Netleştirme işlemi (Orijinal görüntüye kenarları ekleme)
sharp_image = np.clip(I1 + 1.5 * edges - 0.5, 0, 255).astype(np.uint8)

# Sonuçları görselleştiriyoruz
plt.figure(), plt.imshow(I1, cmap='gray')
plt.title('Orijinal Görüntü')

plt.figure(), plt.imshow(sharp_image, cmap='gray')
plt.title('Netleştirilmiş Görüntü (Kenar Tespiti ile)')

plt.show()