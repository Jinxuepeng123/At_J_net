import cv2
from torchvision import transforms
import numpy as np
import os
from PIL import Image
dark_path='D:/picture/'
dark_data_list = os.listdir(dark_path)

for i in range(len(dark_data_list)):
    name = dark_data_list[i][:-4]
    img=cv2.imread(dark_path+name+'.jpg')
    cv2.imwrite('D:/picture2/'+name+'gth'+'.jpg',img)