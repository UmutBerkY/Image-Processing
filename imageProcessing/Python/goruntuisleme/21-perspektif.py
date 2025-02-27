import cv2
import numpy as np
import matplotlib.pyplot as plt

# Görüntüyü yükleyelim
image = cv2.imread('plaka.png')

if image is None:
    print("Dosya yüklenemedi. Lütfen dosya yolunu kontrol edin.")
    exit()

# Şu anki dört nokta (belirttiğiniz)
pts1 = np.float32([[55, 154], [561, 152], [73, 246], [548, 236]])

# Hedef dört nokta (yeni konum)
# Hedef dörtgenin boyutlarını daha net belirtelim
pts2 = np.float32([[0, 0], [250, 0], [0, 266], [250, 266]])

# Perspektif dönüşüm matrisini hesaplayalım
def perspective_transform(pts1, pts2, image):
    height, width = image.shape[:2]
    # Dönüşüm matrisini hesaplayalım
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return M

# Perspektif dönüşümünü uygulayalım
def apply_perspective_transform(M, image, width, height):
    result = cv2.warpPerspective(image, M, (width, height))
    return result

# Perspektif dönüşümünü hesaplayalım
M = perspective_transform(pts1, pts2, image)
result = apply_perspective_transform(M, image, 250, 266)

# Sonuçları görselleştirelim
plt.figure(figsize=(10, 10))

# Orijinal görsel
plt.subplot(1, 2, 1)
plt.imshow(image[:,:,::-1])  # BGR'den RGB'ye dönüşüm matplotlib için
plt.title('Orijinal Görüntü')

# Düzleştirilmiş (perspektif düzeltmesi yapılmış) görsel
plt.subplot(1, 2, 2)
plt.imshow(result[:,:,::-1])  # BGR'den RGB'ye dönüşüm matplotlib için
plt.title('Düzleştirilmiş Görüntü')

plt.show()