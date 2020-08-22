import cv2
from torchvision import transforms
import numpy as np
import os
from PIL import Image
dark_path='/home/liu/jinxuepeng/fivek_orig_jpg1/test_dark/'
dark_gth_path='/home/liu/jinxuepeng/fivek_orig_jpg1/test_gth/'
dark_data_list = os.listdir(dark_path)
dark_gth_list=os.listdir(dark_gth_path)
cut_dark_path='/home/liu/jinxuepeng/test_dark_cut/'
cut_gth_path='/home/liu/jinxuepeng/test_gth_cut/'

if not os.path.exists(cut_dark_path):
    os.makedirs(cut_dark_path)

if not os.path.exists(cut_gth_path):
    os.makedirs(cut_gth_path)

for i in range(len(dark_data_list)):
    name = dark_data_list[i][:-4]
    dark_image = cv2.imread(dark_path + name + '.jpg')
    gth_image = cv2.imread(dark_gth_path + name + '.jpg')
    dark_image=np.array(dark_image)
    gth_image=np.array(gth_image)
    if(dark_image.shape[0]>=1028 and dark_image.shape[1]>=1028):
         dark_image=dark_image[1:1025, 1:1025,: ]
         gth_image = gth_image[1:1025, 1:1025, :]
         cv2.imwrite(cut_dark_path+name+'.jpg',dark_image)
         cv2.imwrite(cut_gth_path+name+'.jpg', gth_image)
         print(i)
