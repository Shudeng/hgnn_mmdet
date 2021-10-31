import torch
import math
from torch_scatter import scatter_max, scatter
from torch_cluster import radius, radius_graph
from torch_cluster import knn_graph

from mmcv.runner import auto_fp16
from torch import nn as nn

from mmdet3d.ops import PointFPModule, build_sa_module
from mmdet.models import BACKBONES
from mmdet3d.models.backbones.base_pointnet import BasePointNet

def multi_layer_neural_network_fn(Ks, relu=True):
    linears = []
    for i in range(1, len(Ks)):
        if relu:
            linears += [
            nn.Linear(Ks[i-1], Ks[i]),
            nn.ReLU(),
            nn.LayerNorm(Ks[i])
#            nn.BatchNorm1d(Ks[i])
            ]
        else:
            linears += [
            nn.Linear(Ks[i-1], Ks[i]),
            nn.LayerNorm(Ks[i])
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
    for i, xyz in enumerate(xyzs):

        edges += [knn_graph(xyz, k)[:, :k*xyz.shape[0]]]
        #print(k*xyz.shape[0], i, xyz.shape, edges[-1].shape)

    return torch.stack(edges) # B x 2 x L, L = (16*n)

class GCN_Module(nn.Module):
    def __init__(self, edge_MLP_depth_list=[256+3, 256], update_MLP_depth_list=[256, 256]):
        super(GCN_Module, self).__init__()
        self.edge_feature_fn = multi_layer_neural_network_fn(edge_MLP_depth_list)
        self.update_fn = multi_layer_neural_network_fn(update_MLP_depth_list)

    def forward(self, xyz, features, edges):
        """
        xys: b x n x 3
        features: b x C x n
        """
        # 
        input_vertex_features = features.permute(0,2,1) # b x n x C
        input_vertex_coordinates = xyz # b x n x 3


        # Gather the source vertex of the edges
        s_vertex_features = torch.gather(input_vertex_features, dim=1, index=edges[:, :, :1].expand(-1, -1, input_vertex_features.shape[-1])) # b x e x C
        s_vertex_coordinates = torch.gather(input_vertex_coordinates, dim=1, index=edges[:, :, :1].expand(-1, -1, 3)) # b x e x 3

        # Gather the destination vertex of the edges
        d_vertex_coordinates = torch.gather(input_vertex_coordinates, dim=1, index=edges[:, :, 1:].expand(-1, -1, 3))

        # Prepare initial edge features
        edge_features = torch.cat([s_vertex_features, s_vertex_coordinates - d_vertex_coordinates], dim=-1) # b x e x (C+3)


        # Extract edge features
        b, e, C = edge_features.shape

        edge_features = edge_features.view(-1, C)
        edge_features = self.edge_feature_fn(edge_features) # b x e x C
        edge_features = edge_features.view(b, e, C-3)

        # Aggregate edge features
        aggregated_edge_features = max_aggregation_fn(edge_features, edges[:, :,1], input_vertex_features.shape[0]) # b x n x C

        # Update vertex features
        b, N, C = aggregated_edge_features.shape
        aggregated_edge_features = aggregated_edge_features.view(-1, C)
        update_features = self.update_fn(aggregated_edge_features) # (bxN) x C
        update_features = update_features.view(b,N,C)

        output_vertex_features  = update_features + input_vertex_features

        return output_vertex_features.permute(0, 2,1).contiguous() # b x C x n

class GAN_Module(nn.Module):
    def __init__(self, edge_MLP_depth_list=[256+3, 256], update_MLP_depth_list=[256, 256], heads=8):
        super(GAN_Module, self).__init__()
#        self.edge_feature_fn = multi_layer_neural_network_fn(edge_MLP_depth_list)
        self.key_fn = multi_layer_neural_network_fn(edge_MLP_depth_list)
        self.value_fn = multi_layer_neural_network_fn(edge_MLP_depth_list)
        self.query_fn = multi_layer_neural_network_fn(edge_MLP_depth_list)
        self.update_fn = multi_layer_neural_network_fn(update_MLP_depth_list)

        assert edge_MLP_depth_list[-1] % heads == 0
        self.heads = heads

    def forward(self, xyz, features, edges):
        """
        xys: b x n x 3
        features: b x C x n
        """
        # 
        input_vertex_features = features.permute(0,2,1) # b x n x C
        input_vertex_coordinates = xyz # b x n x 3


        # Gather the source vertex of the edges
        s_vertex_features = torch.gather(input_vertex_features, dim=1, index=edges[:, :, :1].expand(-1, -1, input_vertex_features.shape[-1])) # b x e x C
        s_vertex_coordinates = torch.gather(input_vertex_coordinates, dim=1, index=edges[:, :, :1].expand(-1, -1, 3)) # b x e x 3

        # Gather the destination vertex of the edges
        d_vertex_features = torch.gather(input_vertex_features, dim=1, index=edges[:, :, 1:].expand(-1, -1, input_vertex_features.shape[-1])) # b x e x C
        d_vertex_coordinates = torch.gather(input_vertex_coordinates, dim=1, index=edges[:, :, 1:].expand(-1, -1, 3)) # b x e x 3

        # Prepare initial edge features
        s_vertex_features = torch.cat([s_vertex_features, s_vertex_coordinates - d_vertex_coordinates], dim=-1)
        d_vertex_features = torch.cat([d_vertex_features, s_vertex_coordinates - d_vertex_coordinates], dim=-1)

        b, e, C = s_vertex_features.shape
        s_vertex_features = s_vertex_features.view(-1, C)
        d_vertex_features = d_vertex_features.view(-1, C)

        key = self.key_fn(s_vertex_features) # (b x e) x C
        key = key.view(b,e, self.heads, -1) # b x e x heads x C

        value = self.value_fn(s_vertex_features) # (b x e) x C
        value = value.view(b,e,self.heads, -1) # b x e x heads x C

        query = self.query_fn(d_vertex_features) # (bxe) x C
        query = query.view(b,e,self.heads, -1) # b x e x heads x C



        # calculate similarity
        simi = (key * query).sum(-1) / math.sqrt(key.shape[-1]) # b x e x heads
        max_ = scatter(simi, edges[:, :, 1], dim=1, reduce="max") # b x N x heads

        max_ = torch.gather(max_, dim=1, index=edges[:,:, 1:].expand(-1,-1,self.heads)) # b x e x heads

        simi = simi-max_ # b x e x heads
        simi = torch.exp(simi) # b x e x heads

        base = scatter(simi, edges[:, :, 1], dim=1, reduce="sum") # b x n x heads
        base = torch.gather(base, dim=1, index=edges[:,:,1:].expand(-1,-1, self.heads)) # b x e x heads
        simi = simi / base # b x e x heads

#        base = scatter(simi, edges[:, :, 1], dim=1, reduce="sum") # b x n x heads
#        print("base", base, base.shape) # base[i, j, k]==1 for all (i,j,k)



        # Aggregate edge features
        value = value * simi[:, :, :, None] # b x e x heads x C
        value = value.view(b, e, -1)


        aggregated_features = scatter(value, edges[:, :, 1], dim=1, reduce="mean") # b x n x C

        # Update vertex features
        b, n, C = aggregated_features.shape
        aggregated_features=aggregated_features.view(-1, C)
        update_features = self.update_fn(aggregated_features) # (bxn) x C
        update_features = update_features.view(b,n,-1) # bxnxC

        output_vertex_features  = update_features + input_vertex_features

        return output_vertex_features.permute(0, 2, 1).contiguous()

class GCN_Block(nn.Module):
    def __init__(self, layers,  edge_MLP_depth_list=[256+3, 256], update_MLP_depth_list=[256, 256]):
        super(GCN_Block, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(layers):
            #self.layers.append(GCN_Module(edge_MLP_depth_list, update_MLP_depth_list))
            self.layers.append(GAN_Module(edge_MLP_depth_list, update_MLP_depth_list))

    def forward(self, xyz, features, edges):
        for i in range(len(self.layers)):
            features = self.layers[i](xyz, features, edges)
        return features

