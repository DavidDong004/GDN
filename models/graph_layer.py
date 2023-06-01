import torch
from torch.nn import Parameter, Linear, Sequential, BatchNorm1d, ReLU
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

from torch_geometric.nn.inits import glorot, zeros
import time
import math

class GraphLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1, concat=True,
                 negative_slope=0.2, dropout=0, bias=True, inter_dim=-1,**kwargs): #in/out_channels为输入输出通道数，head为多头注意力机制头数
        super(GraphLayer, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.__alpha__ = None

        self.lin = Linear(in_channels, heads * out_channels, bias=False) #线性变换层

        self.att_i = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_j = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_em_i = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_em_j = Parameter(torch.Tensor(1, heads, out_channels)) #注意力权重参数

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None) #根据设置（是否有偏置和拼接）确定偏置参数

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin.weight)
        glorot(self.att_i)
        glorot(self.att_j)
        
        zeros(self.att_em_i)
        zeros(self.att_em_j)

        zeros(self.bias) #初始化参数



    def forward(self, x, edge_index, embedding, return_attention_weights=False): #前向传播函数
        """"""
        if torch.is_tensor(x):
            x = self.lin(x)
            x = (x, x)
        else:
            x = (self.lin(x[0]), self.lin(x[1]))

        edge_index, _ = remove_self_loops(edge_index)  #去除连接中包含的自环边 
        edge_index, _ = add_self_loops(edge_index,
                                       num_nodes=x[1].size(self.node_dim)) #在连接中根据节点数量添加自环边

        out = self.propagate(edge_index, x=x, embedding=embedding, edges=edge_index,
                             return_attention_weights=return_attention_weights) #使用消息传递机制更新节点（调用messgae）

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels) #将out转换为形状为batch_size,head*out_channels形状
        else:
            out = out.mean(dim=1) #求在第一个维度上均值

        if self.bias is not None: #加入偏置
            out = out + self.bias

        if return_attention_weights:
            alpha, self.__alpha__ = self.__alpha__, None #清空注意力权重并返回
            return out, (edge_index, alpha)
        else:
            return out

    def message(self, x_i, x_j, edge_index_i, size_i,
                embedding,
                edges,
                return_attention_weights): #消息传递机制更新节点

        x_i = x_i.view(-1, self.heads, self.out_channels) 
        x_j = x_j.view(-1, self.heads, self.out_channels) #修改多头注意力机制，将其作为第二个维度？

        if embedding is not None:
            embedding_i, embedding_j = embedding[edge_index_i], embedding[edges[0]]
            embedding_i = embedding_i.unsqueeze(1).repeat(1,self.heads,1)
            embedding_j = embedding_j.unsqueeze(1).repeat(1,self.heads,1)

            key_i = torch.cat((x_i, embedding_i), dim=-1)
            key_j = torch.cat((x_j, embedding_j), dim=-1)



        cat_att_i = torch.cat((self.att_i, self.att_em_i), dim=-1)
        cat_att_j = torch.cat((self.att_j, self.att_em_j), dim=-1)

        alpha = (key_i * cat_att_i).sum(-1) + (key_j * cat_att_j).sum(-1)


        alpha = alpha.view(-1, self.heads, 1)


        alpha = F.leaky_relu(alpha, self.negative_slope)
        self.node_dim=0
        alpha = softmax(alpha, edge_index_i, num_nodes=size_i)

        if return_attention_weights:
            self.__alpha__ = alpha

        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        return x_j * alpha.view(-1, self.heads, 1)



    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)
