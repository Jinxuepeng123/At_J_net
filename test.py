import sys
import argparse
import time
import glob
from torch.autograd import Variable
import torch
import torch.nn as nn
import os
from torchvision import transforms
from dataloader import AtJDataSet
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from AtJ_model import *
import torch
from utils.loss import *
from utils.save_log_to_excel import *

"""
    测试的具体任务：
    1.测试nyu的测试集，给出可视化结果和指标结果。可以使用excel表格给出指标结果。
    2.测试真实世界的数据集，给出可视化结果。
    3.测试ntire2018数据集，给出可视化结果和指标结果。
"""

# test_visual_path = '/input/data/nyu/test_visual/'

data_path = './input/data/'
test_path = data_path+'nyu/mini_test_dark/'
gth_path = data_path+'nyu/mini_test_gth/'

BATCH_SIZE = 2
weight = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
excel_test_line = 1

def change(image):
    image = image.cpu().numpy()
    image = 1 - image
    #img = np.rollaxis(image, 0, 3)  # 128x128x3
    img = image.astype('float32')
    img=torch.tensor(img)
    #print(img.shape)
    return img.cuda()


def get_image_for_save(img):
    img = img.cpu()
    img = img.numpy()
    img = np.squeeze(img)    #3x128x128
    img = img * 255
    img[img < 0] = 0
    img[img > 255] = 255    #128x128x3
    img = img.astype('uint8')
    print(img.shape)
    return img


save_path = 'test_result_{}'.format(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
if not os.path.exists(save_path):
    os.makedirs(save_path)
excel_save = save_path + '/test_result.xls'

model_path = './AtJ_model/AtJ_model.pt'
net = torch.load(model_path)
net = net.cuda()    #.cuda
transform = transforms.Compose([transforms.ToTensor()])

test_path_list = [test_path, gth_path]
test_data = AtJDataSet(transform, test_path_list, flag='test')
test_data_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

count = 0
print(">>Start testing...\n")
f, sheet_test = init_excel(kind='test')
for name, dark_image, gt_image in test_data_loader:
    count += 1
    print('Processing %d...' % count)
    net.eval()
    with torch.no_grad():
        # J = net(haze_image)
        J, A, t, J_reconstruct, haze_reconstruct = net(change(dark_image))
        loss_image = [J, A, t, gt_image, J_reconstruct, haze_reconstruct, change(dark_image)]
        # for i in range(BATCH_SIZE):
        #    print(i)
        # loss_image = [J, gt_image]
        loss, temp_loss = loss_function(loss_image, weight)
        excel_test_line = write_excel(sheet=sheet_test,
                                      epoch=False,
                                      itr=name[0],
                                      data_type='test',
                                      line=excel_test_line,
                                      loss=temp_loss,
                                      weight=weight)
        f.save(excel_save)
        im_output_for_save = get_image_for_save(J)
        filename = name[0] + '.jpg'
        cv2.imwrite(os.path.join(save_path, filename), im_output_for_save)

print("Finished!")
