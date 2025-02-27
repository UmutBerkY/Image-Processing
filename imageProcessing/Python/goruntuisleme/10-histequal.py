import cv2
import numpy as np
import matplotlib.pyplot as plt

# Görüntüyü okuduk
I1 = cv2.imread('lenna.jpg', cv2.IMREAD_GRAYSCALE)
# Histogram eşitlemeyi manuel olarak uygulama
def equalize_histogram(image):
    # Histogram hesapla
    hist, bins = np.histogram(image.flatten(), bins=256, range=[0, 256])
    # Kümülatif histogram (CDF) hesapla
    cdf = hist.cumsum()
    cdf_normalized = cdf * (255 / cdf[-1])  # Normalizasyon
    # Yeni piksel değerlerini belirleme
    equalized_image = np.interp(image.flatten(), bins[:-1], cdf_normalized)
    return equalized_image.reshape(image.shape).astype(np.uint8)
# Histogram eşitlemeyi uygula
I2 = equalize_histogram(I1)
# Sonuçları görselleştirdik
plt.figure(), plt.imshow(I2, cmap='gray')
plt.title('Histogram Eşitlenmiş Görüntü')
plt.show()

# Histogramı görselleştirdik
plt.figure(), plt.hist(I2.ravel(), bins=256, range=(0, 256), color='black')
plt.title('Histogram Eşitlenmiş Görüntü')
plt.show()
