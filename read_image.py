#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import h5py
import os
from PIL import Image

path = 'D:/OneDrive - Microsoft 365/deep-learning/At_J_net/input/data/nyu/gth'
if not os.path.isdir(path):
    os.makedirs(path)

f = h5py.File("D:/OneDrive - Microsoft 365/deep-learning/At_J_net/nyu_depth_v2_labeled.mat")
images = f["images"]
images = np.array(images)
'''
高最小值为624 （640，8-632）
宽最小值为464 （480，8-472）
'''
images = images[:, :, 8:632, 8:472]
print(images.shape)
'''
for i in range(len(images)):
    print(str(i) + '.bmp')
    r = Image.fromarray(images[i][0]).convert('L')
    g = Image.fromarray(images[i][1]).convert('L')
    b = Image.fromarray(images[i][2]).convert('L')
    img = Image.merge("RGB", (r, g, b))
    save_path = path + str(i) + '.bmp'
    img.save(save_path, optimize=True)
'''
