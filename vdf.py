import cv2
from torchvision import transforms
import numpy as np
from PIL import Image
img=cv2.imread('D:/picture/6.png')



transform = transforms.Compose([transforms.ToTensor()])

img1=transform(img)
img1=img1.numpy()
img2=1-img1
img2=img2*255
img2 = np.rollaxis(img2, 0, 3)
img2 = img2.astype('uint8')
cv2.imshow('kk',img2)
cv2.imshow('kl',img)
cv2.waitKey(0)

