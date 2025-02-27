import cv2
import matplotlib.pyplot as plt

# Görüntüyü yükledik.
griresim = cv2.imread('lenna.jpg', cv2.IMREAD_GRAYSCALE)

# Negatif görüntüyü oluşturduk.
negatif_resim = 255 - griresim

# Sonucu gösterdik.
plt.imshow(negatif_resim, cmap='gray')
plt.title('Negatif Görüntü')
plt.show()
