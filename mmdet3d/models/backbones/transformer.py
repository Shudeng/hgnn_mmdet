import torch
import math
from torch_scatter import scatter_max, scatter
from torch_cluster import radius, radius_graph
from torch_cluster import knn_graph

from mmcv.runner import auto_fp16
from torch import nn as nn

from mmdet3d.ops import PointFPModule, build_sa_module
from mmdet.models import BACKBONES
from .base_pointnet import BasePointNet
import sys
sys.path.insert(0, "/home/shudeng/mmdetection3d")
from downsample_block import DownSampleBlock

def multi_layer_neural_network_fn(Ks):
    linears = []
    for i in range(1, len(Ks)):
        linears += [
        nn.Linear(Ks[i-1], Ks[i]),
        nn.ReLU(),
        nn.LayerNorm(Ks[i])
#        nn.BatchNorm1d(Ks[i])
        ]
    return nn.Sequential(*linears)

def max_aggregation_fn(features, index, l):
    """
    features: b x e x C
    index: b x e
    """
    set_features = scatter(features, index, dim=1, reduce="max")
    return set_features

def build_edge(xyzs, k=16, loop=True):
    """
    xyz: b x n x 3
    """
    edges = []
    for xyz in xyzs:
        edges += [knn_graph(xyz, k)]

    return torch.stack(edges) # B x 2 x L, L = (16*n)

def initialize_weight(x):
    nn.init.xavier_uniform_(x.weight)
    if x.bias is not None:
        nn.init.constant_(x.bias, 0)


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, filter_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(filter_size, hidden_size)

        initialize_weight(self.layer1)
        initialize_weight(self.layer2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, dropout_rate, head_size=8):
        super(MultiHeadAttention, self).__init__()

        self.head_size = head_size

        self.att_size = att_size = hidden_size // head_size
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, head_size * att_size, bias=False)
        self.linear_k = nn.Linear(hidden_size+3, head_size * att_size, bias=False)
        self.linear_v = nn.Linear(hidden_size+3, head_size * att_size, bias=False)
        initialize_weight(self.linear_q)
        initialize_weight(self.linear_k)
        initialize_weight(self.linear_v)

        self.att_dropout = nn.Dropout(dropout_rate)

        self.output_layer = nn.Linear(head_size * att_size, hidden_size,
                                      bias=False)
        initialize_weight(self.output_layer)
        self.heads = head_size
        

    def forward(self, q, k, v, mask=None, cache=None):
        """
        k, v : b x n x e_n x C
        q: b x n x C


        """
        b, n, e_n, C = k.shape

        k = self.linear_k(k) # (b x e) x C
        v = self.linear_v(v) # (b x e) x C
        q = self.linear_q(q)

        k = k.view(b, n, e_n, self.heads, -1) 
        v = v.view(b, n, e_n, self.heads, -1) 
        q = q.view(b, n, 1, self.heads, -1) 

        # calculate similarity
        #simi = (key * query).sum(-1) / math.sqrt(key.shape[-1]) # b x e x heads
        simi = (k * q).sum(-1) / math.sqrt(k.shape[-1]) # b x n x e_n x self.heads x 1
        simi_softmax = nn.functional.softmax(simi, dim=2) # b x n x e_n x self.heads 
        simi_softmax = self.att_dropout(simi_softmax)

        v = v * simi_softmax[:, :, :, :, None] # b x n x e_n x self.heads x C

        v = v.sum(2) # b x n x self.heads x C
        v = v.view(b,n,-1)
        v = self.output_layer(v)
        return v


class GAN_Module(nn.Module):
    def __init__(self, hidden_size=256, filter_size=512, dropout_rate=0.1, heads=8):
        super(GAN_Module, self).__init__()
        self.self_attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.self_attention = MultiHeadAttention(hidden_size, dropout_rate)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn = FeedForwardNetwork(hidden_size, filter_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

        self.downsample_edges = 8
        self.downsample_layer = nn.Sequential(*[
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size//2, self.downsample_edges)
            ])

        self.heads = heads

    def forward(self, xyz, features, edges):
        """
        xys: b x n x 3
        features: b x C x n
        """

        """
        print("edges", edges)
        print("edges.shape", edges.shape)
        for i in range(edges.shape[0]):
            for j in range(edges.shape[1]):
                assert edges[i][j][1]  == j//16
                print(edges[i][j][1], j//16)
        """

        features = features.permute(0,2,1) # b x n x C
        input_vertex_features = self.self_attention_norm(features)

        input_vertex_coordinates = xyz # b x n x 3
        b, n, C = input_vertex_features.shape
        e = edges.shape[1]
        # Gather the source vertex of the edges
        s_vertex_features = torch.gather(input_vertex_features, dim=1, index=edges[:, :, :1].expand(-1, -1, input_vertex_features.shape[-1])) # b x e x C
        s_vertex_coordinates = torch.gather(input_vertex_coordinates, dim=1, index=edges[:, :, :1].expand(-1, -1, 3)) # b x e x 3
        s_vertex_features = s_vertex_features.view(b, n, e//n, -1)
        s_vertex_coordinates = s_vertex_coordinates.view(b, n, e//n, -1)

        b, n, e_n, C = s_vertex_features.shape

        # downsample edges from e//n to e'
        # gumbel softmax
        downsample_w = self.downsample_layer(s_vertex_features.view(b*n, e//n, -1)) # (bxn) x (e//n) x e', e'=8
        downsample_w = downsample_w.permute(0,2,1) # (bxn) x e' x (e//n)
        downsample_w = nn.functional.gumbel_softmax(downsample_w, tau=0.1)

        s_vertex_features = torch.bmm(downsample_w, s_vertex_features.view(b*n, e//n, -1)).view(b, n, -1, C) # b x n x e' x C
        s_vertex_coordinates = torch.bmm(downsample_w, s_vertex_coordinates.view(b*n, e//n, -1)).view(b, n, -1, 3) # b x n x e' x 3
        s_vertex_features = torch.cat([s_vertex_features, 
            s_vertex_coordinates - input_vertex_coordinates[:, :, None, :]
            ], dim=-1) # b, n, e//n, C



        input_vertex_features = self.self_attention(input_vertex_features, s_vertex_features, s_vertex_features)
        input_vertex_features = self.self_attention_dropout(input_vertex_features)
        features = features + input_vertex_features

        input_vertex_features = self.ffn_norm(features)
        input_vertex_features = self.ffn(input_vertex_features)
        input_vertex_features = self.ffn_dropout(input_vertex_features)
        features = features + input_vertex_features
        return features.permute(0, 2, 1).contiguous()


class GCN_Block(nn.Module):
    def __init__(self, layers,  edge_MLP_depth_list=[256+3, 256], update_MLP_depth_list=[256, 256]):
        super(GCN_Block, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(layers):
            #self.layers.append(GCN_Module(edge_MLP_depth_list, update_MLP_depth_list))
            self.layers.append(GAN_Module())

    def forward(self, xyz, features, edges):
        for i in range(len(self.layers)):
            features = self.layers[i](xyz, features, edges)
        return features

