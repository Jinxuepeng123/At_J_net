import cv2
from torchvision import transforms
import numpy as np
from PIL import Image
img=cv2.imread('D:/picture/9.png')




img2 = np.rollaxis(img2, 0, 3)
img2 = img2.astype('uint8')
cv2.imshow('kk',img2)
cv2.waitKey(0)

