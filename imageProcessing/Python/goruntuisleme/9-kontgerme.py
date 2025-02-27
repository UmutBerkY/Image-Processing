import numpy as np
import cv2

# Görüntüyü yükle
I1 = 'lenna.PNG'
image = cv2.imread(I1, cv2.IMREAD_GRAYSCALE)  # Gri tonlamalı olarak yükle

# Orijinal görüntüyü göster
cv2.imshow('Orijinal Goruntu', image)

# Görüntü 15-38 aralığında renklendirilmiş, bu aralığı 0-255'e gerelim
min_val = 15
max_val = 38

# Normalize etme işlemi
duzeltilmis = ((image - min_val) / (max_val - min_val) * 255).clip(0, 255).astype(np.uint8)

# Sonucu göster
cv2.imshow('Kontrast Germe Sonucu', duzeltilmis)
cv2.waitKey(0)
cv2.destroyAllWindows()
