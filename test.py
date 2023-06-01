import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import time
from util.time import *
from util.env import *

import argparse
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.nn.functional as F


from util.data import *
from util.preprocess import *



def test(model, dataloader):
    # test
   # loss_func = nn.MSELoss(reduction='mean') #损失函数MSE
    loss_func=nn.SmoothL1Loss(reduction='mean',beta = 1.0) #损失函数Huber
    device = get_device() #获取设备信息

    test_loss_list = []#存储测试样本损失值
    now = time.time()#获取时间

    test_predicted_list = []
    test_ground_list = []
    test_labels_list = []#存储测试结果

    t_test_predicted_list = []
    t_test_ground_list = []
    t_test_labels_list = []#存储每个批次的测试结果

    test_len = len(dataloader)#获取dataloader长度（其实就是x的长度）

    model.eval()#设置模型为评估模式

    i = 0
    acu_loss = 0
    for x, y, labels, edge_index in dataloader:
        x, y, labels, edge_index = [item.to(device).float() for item in [x, y, labels, edge_index]] #项数据传入给设备
        
        with torch.no_grad():#禁用梯度计算
            predicted = model(x, edge_index).float().to(device)  #计算预测值
            
            
            loss = loss_func(predicted, y) #计算损失值
            

            labels = labels.unsqueeze(1).repeat(1, predicted.shape[1]) #对label进行变形，使其与batch_num x noden_num 相吻合

            if len(t_test_predicted_list) <= 0:
                t_test_predicted_list = predicted
                t_test_ground_list = y
                t_test_labels_list = labels
            else:
                t_test_predicted_list = torch.cat((t_test_predicted_list, predicted), dim=0)
                t_test_ground_list = torch.cat((t_test_ground_list, y), dim=0)
                t_test_labels_list = torch.cat((t_test_labels_list, labels), dim=0)
        
        test_loss_list.append(loss.item())
        acu_loss += loss.item()
        
        i += 1

        if i % 10000 == 1 and i > 1:
            print(timeSincePlus(now, i / test_len))


    test_predicted_list = t_test_predicted_list.tolist()        
    test_ground_list = t_test_ground_list.tolist()        
    test_labels_list = t_test_labels_list.tolist()      
    
    avg_loss = sum(test_loss_list)/len(test_loss_list)

    return avg_loss, [test_predicted_list, test_ground_list, test_labels_list]




