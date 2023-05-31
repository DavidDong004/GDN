import torch
from torch.utils.data import Dataset, DataLoader

import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np


class TimeDataset(Dataset):
    def __init__(self, raw_data, edge_index, mode='train', config = None): #输入 原始处理完成的数据、连接关系、类型、划分配置
        self.raw_data = raw_data

        self.config = config
        self.edge_index = edge_index
        self.mode = mode

        x_data = raw_data[:-1]#提取出数据矩阵nxT
        labels = raw_data[-1]#提取出标签矩阵1xT


        data = x_data

        # to tensor
        data = torch.tensor(data).double()
        labels = torch.tensor(labels).double()#转换为tensor

        self.x, self.y, self.labels = self.process(data, labels)#将已经获得的数据根据配置转换为需要的数据格式
    
    def __len__(self):
        return len(self.x)#将DataSet的类的长度定义被预测x的切片长度


    def process(self, data, labels):
        x_arr, y_arr = [], []
        labels_arr = []

        slide_win, slide_stride = [self.config[k] for k
            in ['slide_win', 'slide_stride']
        ]
        is_train = self.mode == 'train' #判定状态标签

        node_num, total_time_len = data.shape #提取出特征数量n和时间长度T

        rang = range(slide_win, total_time_len, slide_stride) if is_train else range(slide_win, total_time_len) #根据间隔对数据进行提取（循环间隔）(训练数据有间隔，测试数据没有)
        
        for i in rang:
            #以下两句是原文公式(4)

            ft = data[:, i-slide_win:i] #对现有数据进行切片，让其变成长度为slide_win的连续窗口数据 ，是x之前的
            tar = data[:, i] #获取预测数据，也就是目标预测函数y

            x_arr.append(ft)
            y_arr.append(tar)

            labels_arr.append(labels[i])#存储相对应的数据


        x = torch.stack(x_arr).contiguous()
        y = torch.stack(y_arr).contiguous()#将数据保持形状变为张量，并在内存上进行顺序存储，提高运算速度

        labels = torch.Tensor(labels_arr).contiguous() #将数据在内存上进行顺序存储
        
        return x, y, labels

    def __getitem__(self, idx): #在获取[idx]的时候，获取的是相应的x，y，label和所对应的

        feature = self.x[idx].double()
        y = self.y[idx].double()

        edge_index = self.edge_index.long()

        label = self.labels[idx].double()

        return feature, y, label, edge_index





