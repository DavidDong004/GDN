import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv
from dgl.utils import add_self_loop

class GraphLayer(nn.Module):
    def __init__(self, in_channels, out_channels, heads=1, concat=True,
                 negative_slope=0.2, dropout=0, bias=True, inter_dim=-1):
        super(GraphLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)

        self.att_i = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_j = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_em_i = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_em_j = nn.Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_i)
        nn.init.xavier_uniform_(self.att_j)
        nn.init.zeros_(self.att_em_i)
        nn.init.zeros_(self.att_em_j)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, g, x, embedding=None, return_attention_weights=False):
        x = self.lin(x).view(-1, self.heads, self.out_channels)

        g = add_self_loop(g)

        g.ndata['x'] = x
        g.edata['embedding'] = embedding

        g.update_all(self.message_func, self.reduce_func)

        x = g.ndata.pop('x')
        alpha = g.edata.pop('alpha')

        if return_attention_weights:
            return x.view(-1, self.heads * self.out_channels), alpha
        else:
            return x.mean(dim=1)

    def message_func(self, edges):
        x_i = edges.src['x']
        x_j = edges.dst['x']
        embedding_i = edges.data['embedding'][edges.srcidx].unsqueeze(1)
        embedding_j = edges.data['embedding'][edges.dstidx].unsqueeze(1)

        key_i = torch.cat((x_i, embedding_i), dim=-1)
        key_j = torch.cat((x_j, embedding_j), dim=-1)

        cat_att_i = torch.cat((self.att_i, self.att_em_i), dim=-1)
        cat_att_j = torch.cat((self.att_j, self.att_em_j), dim=-1)

        alpha = (key_i * cat_att_i).sum(dim=-1) + (key_j * cat_att_j).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = F.softmax(alpha, dim=1)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return {'x_j': x_j, 'alpha': alpha}

    def reduce_func(self, nodes):
        x_j = nodes.mailbox['x_j']
        alpha = nodes.mailbox['alpha']

        x = x_j * alpha.unsqueeze(-1)
        return {'x': x}

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)