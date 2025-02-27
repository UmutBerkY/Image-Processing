import cv2
import matplotlib.pyplot as plt

# Gri resimi kullandık.
griresim = cv2.imread('lenna.jpg')

# Histogramı hesapladık.
plt.hist(griresim.ravel(), bins=256, range=[0, 256])
plt.title('Gri Tonlama Görüntü Histogramı')
plt.xlabel('Piksel Değeri')
plt.ylabel('Frekans')
plt.show()
