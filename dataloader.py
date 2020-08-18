from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import pickle
import os
import cv2
import scipy.io as sio
import torch


# 一次性读入所有数据
# nyu/test/1318_a=0.55_b=1.21.png
class AtJDataSet(Dataset):
    def __init__(self, transform1, path=None, flag='train'):
        # print(path)
        self.flag = flag
        self.transform1 = transform1
        self.dark_path, self.gt_path=path
        self.dark_data_list = os.listdir(self.dark_path)
        self.gt_data_list = os.listdir(self.gt_path)
        self.gt_data_list.sort(key=lambda x: int(x[:-4]))
        #print(self.gt_data_list)
        self.dark_data_list.sort(key=lambda x: int(x[:-4]))
        self.length = len(os.listdir(self.dark_path))
        self.dark_image_dict = {}
        self.gth_image_dict = {}
        # 读入数据
        print('starting read image data...')
        for i in range(len(self.dark_data_list)):
            name = self.dark_data_list[i][:-4]
            # print(self.haze_path + name + '.png')
            self.dark_image_dict[name] = cv2.imread(self.dark_path + name + '.jpg')
            # print(self.haze_image_dict[name][0][0][0])
        print('starting read GroundTruth data...')
        for i in range(len(self.gt_data_list)):
            name = self.gt_data_list[i][:-4]
            self.gth_image_dict[name] = cv2.imread(self.gt_path + name + '.jpg')

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        name = self.dark_data_list[idx][:-4]
        dark_image = self.dark_image_dict[name]
        gt_image = self.gth_image_dict[name]
        # print(haze_image[0][0][0])
        # print(gt_image[0][0][0])
        # print(A_gth[0][0])
        # print(t_gth[0][0])

        if self.transform1:
            dark_image = self.transform1(dark_image)
            gt_image = self.transform1(gt_image)

        dark_image = dark_image.cuda()
        gt_image = gt_image.cuda()
        if self.flag == 'train':

            return name, 1-dark_image, gt_image
        elif self.flag == 'test':
            return name, 1-dark_image, gt_image

        # if __name__ == '__main__':
