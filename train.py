# -*- coding: utf-8 -*-
# git clone https://github.com/zhanglideng/At_J_net.git


import sys

# sys.path.append('/home/aistudio/external-libraries')
import os

# if not os.path.exists('/home/aistudio/.torch/models/vgg16-397923af.pth'):
#    os.system('mkdir /home/aistudio/.torch')
#    os.system('mkdir /home/aistudio/.torch/models')
#    os.system('cp /home/aistudio/work/pre_model/*  /home/aistudio/.torch/models/')
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from utils.loss import *
from utils.print_time import *
from utils.save_log_to_excel import *
from dataloader import AtJDataSet
from AtJ_model import *
import time
import xlwt
from utils.ms_ssim import *

LR = 0.00008  # 学习率
EPOCH = 40  # 轮次
BATCH_SIZE = 1  # 批大小
excel_train_line = 1  # train_excel写入的行的下标
excel_val_line = 1  # val_excel写入的行的下标
alpha = 1  # 损失函数的权重
accumulation_steps = 8  # 梯度积累的次数，类似于batch-size=64
# itr_to_lr = 10000 // BATCH_SIZE  # 训练10000次后损失下降50%
itr_to_excel = 8 // BATCH_SIZE  # 训练64次后保存相关数据到excel

weight = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.0001, 0.0001]
loss_num = len(weight)  # 包括参加训练和不参加训练的loss

data_path = './input/data/'
train_dark_path = data_path + 'nyu/mini_train_dark/'  # 暗图像训练集的路径
val_dark_path = data_path + 'nyu/mini_val_dark/'  # 暗图像验证集的路径
train_gt_path = data_path + 'nyu/mini_train_gth/'      #暗图像训练集的gth
val_gt_path=data_path+'nyu/mini_val_gth/'      #暗图像验证集的gth

save_path = './AtJ_result_nyu_' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '/'
save_model_name = save_path + 'AtJ_model.pt'  # 保存模型的路径
excel_save = save_path + 'result.xls'  # 保存excel的路径
mid_save_ed_path = './AtJ_model/AtJ_model.pt'  # 保存的中间模型，用于下一步训练。

# 初始化excel
f, sheet_train, sheet_val = init_excel(kind='train')

if os.path.exists('./AtJ_model/AtJ_model.pt'):
    net = torch.load('./AtJ_model/AtJ_model.pt').cuda()
else:
    net = AtJ().cuda()     #.cuda

if not os.path.exists(save_path):
    os.makedirs(save_path)

# 数据转换模式
transform = transforms.Compose([transforms.ToTensor()])   #用来归一化的
# 读取训练集数据
train_path_list = [train_dark_path,train_gt_path ]
train_data = AtJDataSet(transform, train_path_list)
train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
#print(train_data_loader)
# 读取验证集数据
val_path_list = [val_dark_path, val_gt_path]
val_data = AtJDataSet(transform, val_path_list)
val_data_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# 定义优化器
optimizer = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=1e-5)

#for name, param in net.named_parameters(): #查看可优化的参数有哪些
  #if param.requires_grad:
    #print(name)

min_loss = 999999999
min_epoch = 0
itr = 0
start_time = time.time()

# 开始训练
print("\nstart to train!")
for epoch in range(EPOCH):
    index = 0
    train_loss = 0
    loss_excel = [0] * loss_num
    net.train()
    for name, dark_pre_image, gt_image in train_data_loader:
        index += 1
        itr += 1
        J, A, t, J_reconstruct, dark_reconstruct = net(dark_pre_image)
        dark_reconstruct_real = (1-gt_image)*t+A*(1-t)
        # J, A, t = net(haze_image)
        loss_image = [J, A, t, gt_image, J_reconstruct, dark_reconstruct, dark_pre_image,dark_reconstruct_real]
        loss, temp_loss = loss_function(loss_image, weight)
        train_loss += loss.item()
        loss_excel = [loss_excel[i] + temp_loss[i] for i in range(len(loss_excel))]
        loss = loss / accumulation_steps
        loss.backward()
        # 3. update parameters of net
        if ((index) % accumulation_steps) == 0:
            # optimizer the net
            optimizer.step()  # update parameters of net
            optimizer.zero_grad()  # reset gradient
        if np.mod(index, itr_to_excel) == 0:      #itr_to_excel
            loss_excel = [loss_excel[i] / itr_to_excel for i in range(len(loss_excel))]
            print('epoch %d, %03d/%d' % (epoch + 1, index, len(train_data_loader)))
            print('J_L2=%.5f\n' 'J_SSIM=%.5f\n' 'J_VGG=%.5f\n'
                  'J_re_L2=%.5f\n' 'J_re_SSIM=%.5f\n' 'J_re_VGG=%.5f\n''haze_re_l2_loss=%.5f\n' 'haze_re_ssim_loss=%.5f\n' 'haze_re_vgg_loss=%.5f\n'
                  'haze_re_real_l2_loss=%.5f\n' 'haze_re_real_ssim_loss=%.5f\n' 'haze_re_real_vgg_loss=%.5f\n' 
                  'J_color_loss=%.5f\n' 'J_re_color_loss=%.5f\n' 'losssum=%.5f\n' 'losscurrent=%.5f\n'
                  % (loss_excel[0], loss_excel[1], loss_excel[2], loss_excel[3], loss_excel[4], loss_excel[5],
                     loss_excel[6], loss_excel[7], loss_excel[8], loss_excel[9], loss_excel[10], loss_excel[11],
                     loss_excel[12], loss_excel[13],train_loss,loss))
            # print('L2=%.5f\n' 'SSIM=%.5f\n' % (loss_excel[0], loss_excel[1]))
            print_time(start_time, index, EPOCH, len(train_data_loader), epoch)
            excel_train_line = write_excel(sheet=sheet_train,
                                           data_type='train',
                                           line=excel_train_line,
                                           epoch=epoch,
                                           itr=itr,
                                           loss=loss_excel,
                                           weight=weight)
            f.save(excel_save)
            loss_excel = [0] * loss_num
    optimizer.step()
    optimizer.zero_grad()
    print("A", A)
    print("t", t)
    print("J", J)
    print("gt_image", gt_image)
    print("J_reconstruct", J_reconstruct)
    print("dark_reconstruct", dark_reconstruct)
    print("dark_pre_image", dark_pre_image)
    print("dark_reconstruct_real", dark_reconstruct_real)
    loss_excel = [0] * loss_num
    val_loss = 0
    with torch.no_grad():
        net.eval()
        for name,dark_pre_image, gt_image in val_data_loader:
            J, A, t, J_reconstruct, dark_reconstruct = net(dark_pre_image)
            loss_image = [J, A, t, gt_image,J_reconstruct, dark_reconstruct, dark_pre_image]
            loss, temp_loss = loss_function(loss_image, weight)

            loss_excel = [loss_excel[i] + temp_loss[i] for i in range(len(loss_excel))]
    train_loss = train_loss / len(train_data_loader)
    loss_excel = [loss_excel[i] / len(val_data_loader) for i in range(len(loss_excel))]
    for i in range(len(loss_excel)):
        val_loss = val_loss + loss_excel[i] * weight[i]
    print('J_L2=%.5f\n' 'J_SSIM=%.5f\n' 'J_VGG=%.5f\n'
          'J_re_L2=%.5f\n' 'J_re_SSIM=%.5f\n' 'J_re_VGG=%.5f\n'
          % (loss_excel[0], loss_excel[1], loss_excel[2], loss_excel[3], loss_excel[4], loss_excel[5]))
    # print('L2=%.5f\n' 'SSIM=%.5f\n' % (loss_excel[0], loss_excel[1]))
    excel_val_line = write_excel(sheet=sheet_val,
                                 data_type='val',
                                 line=excel_val_line,
                                 epoch=epoch,
                                 itr=False,
                                 loss=[loss_excel, val_loss, train_loss],
                                 weight=False)
    f.save(excel_save)
   #print('val_loss=',val_loss)
    if val_loss < min_loss:
        min_loss = val_loss
        min_epoch = epoch
        torch.save(net, save_model_name)
        torch.save(net, mid_save_ed_path)
        print('saving the epoch %d model with %.5f' % (epoch + 1, min_loss))
print('Train is Done!')