# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split, Subset

from sklearn.preprocessing import MinMaxScaler

from util.env import get_device, set_device
from util.preprocess import build_loc_net, construct_data
from util.net_struct import get_feature_map, get_fc_graph_struc
from util.iostream import printsep

from datasets.TimeDataset import TimeDataset


from models.GDN import GDN

from train import train
from test  import test
from evaluate import get_err_scores, get_best_performance_data, get_val_performance_data, get_full_err_scores

import sys
from datetime import datetime

import os
import argparse
from pathlib import Path

import matplotlib.pyplot as plt

import json
import random

class Main():
    def __init__(self, train_config, env_config, debug=False):

        self.train_config = train_config #传递训练参数
        self.env_config = env_config #传递环境参数
        self.datestr = None #数据指针

        dataset = self.env_config['dataset'] 
        train_orig = pd.read_csv(f'./data/{dataset}/train.csv', sep=',', index_col=0)
        test_orig = pd.read_csv(f'./data/{dataset}/test.csv', sep=',', index_col=0) #读取数据
       
        train, test = train_orig, test_orig #数据赋值

        if 'attack' in train.columns:
            train = train.drop(columns=['attack']) #去掉标签，将其视为无监督学习

        feature_map = get_feature_map(dataset) #获取标签数据/获取存在的多元时间序列每个序列的名称的list
        fc_struc = get_fc_graph_struc(dataset) #构建连接关系，一个序列对一个序列的连接关系

        set_device(env_config['device'])
        self.device = get_device() #测试使用，检测使用设备(GPU/CPU)

        fc_edge_index = build_loc_net(fc_struc, list(train.columns), feature_map=feature_map) #建图，根据上文提取的标签数据和连接关系
        fc_edge_index = torch.tensor(fc_edge_index, dtype = torch.long) #加入torch(2xnx(n-1)) 第0行是每个结点的联写节点，第一行是每个结点

        self.feature_map = feature_map

        train_dataset_indata = construct_data(train, feature_map, labels=0)#构建训练数据集（零监督）将原始数据集变成n+1行t列的数据，最后一行为异常标签
        test_dataset_indata = construct_data(test, feature_map, labels=test.attack.tolist())#构建测试数据集


        cfg = {
            'slide_win': train_config['slide_win'],
            'slide_stride': train_config['slide_stride'],
        } #时序数据参数

        train_dataset = TimeDataset(train_dataset_indata, fc_edge_index, mode='train', config=cfg) #将数据根据相关参数(win大小 stride长度)和现有特征连接图对数据进行处理，变为数据为这个包括三部分，训练数据x为win X个数，预测数据y为训练数据个数，异常标签label
        test_dataset = TimeDataset(test_dataset_indata, fc_edge_index, mode='test', config=cfg)#将数据根据相关参数(win大小 stride长度)和现有特征连接图对数据进行处理


        train_dataloader, val_dataloader = self.get_loaders(train_dataset, train_config['seed'], train_config['batch'], val_ratio = train_config['val_ratio']) #将训练数据集变为训练与验证数据集，并进行切片

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset #载入已经处理后的数据（将原始数据分割为x（用来预测的数据），y(预测数据),，label(数据异常标记)）


        self.train_dataloader = train_dataloader 
        self.val_dataloader = val_dataloader #将加载后的数据进类
        self.test_dataloader = DataLoader(test_dataset, batch_size=train_config['batch'],
                            shuffle=False, num_workers=0) #将测试数据集按batch进行划分


        edge_index_sets = []
        edge_index_sets.append(fc_edge_index) #节点与节点之间的连间关系（首先假设都有关系）对长度进行了压缩，将原本为2xnx（n-1）的数组变成了1维的list，这就是为什么要用append

        self.model = GDN(edge_index_sets, len(feature_map), 
                dim=train_config['dim'], 
                input_dim=train_config['slide_win'],
                out_layer_num=train_config['out_layer_num'],
                out_layer_inter_dim=train_config['out_layer_inter_dim'],
                topk=train_config['topk']
            ).to(self.device)#将数据导入GDN模型中（包括连接关系和参数，这得进去看）



    def run(self):

        if len(self.env_config['load_model_path']) > 0:
            model_save_path = self.env_config['load_model_path']
        else:
            model_save_path = self.get_save_path()[0]#对训练模型进行存储

            self.train_log = train(self.model, model_save_path, 
                config = train_config,
                train_dataloader=self.train_dataloader,
                val_dataloader=self.val_dataloader, 
                feature_map=self.feature_map,
                test_dataloader=self.test_dataloader,
                test_dataset=self.test_dataset,
                train_dataset=self.train_dataset,
                dataset_name=self.env_config['dataset']
            ) #训练模型
        
        # test            
        self.model.load_state_dict(torch.load(model_save_path))
        best_model = self.model.to(self.device) #对模型进行读取

        _, self.test_result = test(best_model, self.test_dataloader)
        _, self.val_result = test(best_model, self.val_dataloader) #对预测阶段进行评估

        self.get_score(self.test_result, self.val_result) #获取异常分数

    def get_loaders(self, train_dataset, seed, batch, val_ratio=0.1): #创建训练集和验证集
        dataset_len = int(len(train_dataset)) #计算数据集长度
        train_use_len = int(dataset_len * (1 - val_ratio))#计算训练集长度
        val_use_len = int(dataset_len * val_ratio) #计算验证集长度
        val_start_index = random.randrange(train_use_len) #随机选择验证集起始索引
        indices = torch.arange(dataset_len)#创建索引张量

        train_sub_indices = torch.cat([indices[:val_start_index], indices[val_start_index+val_use_len:]])#训练索引子数据集切片，合并训练集长度
        train_subset = Subset(train_dataset, train_sub_indices)#给训练数据集作切片

        val_sub_indices = indices[val_start_index:val_start_index+val_use_len] #验证子数据集索引
        val_subset = Subset(train_dataset, val_sub_indices)#给验证数据集切片


        train_dataloader = DataLoader(train_subset, batch_size=batch,
                                shuffle=True)

        val_dataloader = DataLoader(val_subset, batch_size=batch,
                                shuffle=False)

        return train_dataloader, val_dataloader 

    def get_score(self, test_result, val_result):#获取模型的性能指标

        feature_num = len(test_result[0][0]) #节点数
        np_test_result = np.array(test_result) #测试数据
        np_val_result = np.array(val_result) #验证数据

        test_labels = np_test_result[2, :, 0].tolist() #获取测试数据标签
    
        test_scores, normal_scores = get_full_err_scores(test_result, val_result)

        top1_best_info = get_best_performance_data(test_scores, test_labels, topk=1)  #获取测试集最佳指标
        top1_val_info = get_val_performance_data(test_scores, normal_scores, test_labels, topk=1) #获取验证集最佳指标


        print('=========================** Result **============================\n')

        info = None
        if self.env_config['report'] == 'best':
            info = top1_best_info
        elif self.env_config['report'] == 'val':
            info = top1_val_info

        print(f'F1 score: {info[0]}')
        print(f'precision: {info[1]}')
        print(f'recall: {info[2]}') #打印结果
        print(f'auc:{info[3]}')
        print(f"thresold:{info[4]}\n")

    def get_save_path(self, feature_name=''):#存储模型

        dir_path = self.env_config['save_path']
        
        if self.datestr is None:
            now = datetime.now()
            self.datestr = now.strftime('%m|%d-%H:%M:%S')
        datestr = self.datestr          

        paths = [
            f'./pretrained/{dir_path}/best_{datestr}.pt',
            f'./results/{dir_path}/{datestr}.csv',
        ]

        for path in paths:
            dirname = os.path.dirname(path)
            Path(dirname).mkdir(parents=True, exist_ok=True)

        return paths

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-batch', help='batch size', type = int, default=128)
    parser.add_argument('-epoch', help='train epoch', type = int, default=100)
    parser.add_argument('-slide_win', help='slide_win', type = int, default=15)
    parser.add_argument('-dim', help='dimension', type = int, default=64)
    parser.add_argument('-slide_stride', help='slide_stride', type = int, default=5)
    parser.add_argument('-save_path_pattern', help='save path pattern', type = str, default='')
    parser.add_argument('-dataset', help='wadi / swat', type = str, default='wadi')
    parser.add_argument('-device', help='cuda / cpu', type = str, default='cuda')
    parser.add_argument('-random_seed', help='random seed', type = int, default=0)
    parser.add_argument('-comment', help='experiment comment', type = str, default='')
    parser.add_argument('-out_layer_num', help='outlayer num', type = int, default=1)
    parser.add_argument('-out_layer_inter_dim', help='out_layer_inter_dim', type = int, default=256)
    parser.add_argument('-decay', help='decay', type = float, default=0)
    parser.add_argument('-val_ratio', help='val ratio', type = float, default=0.1)
    parser.add_argument('-topk', help='topk num', type = int, default=20)
    parser.add_argument('-report', help='best / val', type = str, default='best')
    parser.add_argument('-load_model_path', help='trained model path', type = str, default='')

    args = parser.parse_args()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(args.random_seed)


    train_config = {
        'batch': args.batch,
        'epoch': args.epoch,
        'slide_win': args.slide_win,
        'dim': args.dim,
        'slide_stride': args.slide_stride,
        'comment': args.comment,
        'seed': args.random_seed,
        'out_layer_num': args.out_layer_num,
        'out_layer_inter_dim': args.out_layer_inter_dim,
        'decay': args.decay,
        'val_ratio': args.val_ratio,
        'topk': args.topk,
    }

    env_config={
        'save_path': args.save_path_pattern,
        'dataset': args.dataset,
        'report': args.report,
        'device': args.device,
        'load_model_path': args.load_model_path
    }
    

    main = Main(train_config, env_config, debug=False)
    main.run()





