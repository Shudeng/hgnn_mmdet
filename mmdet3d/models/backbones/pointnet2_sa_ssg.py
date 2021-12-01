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
from .transformer import GCN_Block
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

@BACKBONES.register_module()
class PointNet2SASSG(BasePointNet):
    """PointNet2 with Single-scale grouping.

    Args:
        in_channels (int): Input channels of point cloud.
        num_points (tuple[int]): The number of points which each SA
            module samples.
        radius (tuple[float]): Sampling radii of each SA module.
        num_samples (tuple[int]): The number of samples for ball
            query in each SA module.
        sa_channels (tuple[tuple[int]]): Out channels of each mlp in SA module.
        fp_channels (tuple[tuple[int]]): Out channels of each mlp in FP module.
        norm_cfg (dict): Config of normalization layer.
        sa_cfg (dict): Config of set abstraction module, which may contain
            the following keys and values:

            - pool_mod (str): Pool method ('max' or 'avg') for SA modules.
            - use_xyz (bool): Whether to use xyz as a part of features.
            - normalize_xyz (bool): Whether to normalize xyz with radii in
              each SA module.
    """

    def __init__(self,
                 in_channels,
                 num_points=(2048, 1024, 512, 256),
                 radius=(0.2, 0.4, 0.8, 1.2),
                 num_samples=(64, 32, 16, 16),
                 sa_channels=((64, 64, 128), (128, 128, 256), (128, 128, 256),
                              (128, 128, 256)),
                 fp_channels=((256, 256), (256, 256)),
                 norm_cfg=dict(type='BN2d'),
                 sa_cfg=dict(
                     type='PointSAModule',
                     pool_mod='max',
                     use_xyz=True,
                     normalize_xyz=True),
                 gcn_cfg=[3,3,3]):
        super().__init__()
        self.num_sa = len(sa_channels)
        self.num_fp = len(fp_channels)

        assert len(num_points) == len(radius) == len(num_samples) == len(
            sa_channels)
        assert len(sa_channels) >= len(fp_channels)

        self.SA_modules = nn.ModuleList()
        sa_in_channel = in_channels - 3  # number of channels without xyz
        skip_channel_list = [sa_in_channel]

        for sa_index in range(self.num_sa):
            cur_sa_mlps = list(sa_channels[sa_index])
            cur_sa_mlps = [sa_in_channel] + cur_sa_mlps
            sa_out_channel = cur_sa_mlps[-1]

            self.SA_modules.append(
                build_sa_module(
                    num_point=num_points[sa_index],
                    radius=radius[sa_index],
                    num_sample=num_samples[sa_index],
                    mlp_channels=cur_sa_mlps,
                    norm_cfg=norm_cfg,
                    cfg=sa_cfg,
                    downsample_block=DownSampleBlock(
                        sa_channels[sa_index-1][-1], 
                        sa_channels[sa_index][-1],
                        num_points[sa_index-1],
                        num_points[sa_index]
                        ) if sa_index!=0 else None
                    ))
            skip_channel_list.append(sa_out_channel)
            sa_in_channel = sa_out_channel

        self.FP_modules = nn.ModuleList()

        fp_source_channel = skip_channel_list.pop()
        fp_target_channel = skip_channel_list.pop()
        for fp_index in range(len(fp_channels)):
            cur_fp_mlps = list(fp_channels[fp_index])
            cur_fp_mlps = [fp_source_channel + fp_target_channel] + cur_fp_mlps
            self.FP_modules.append(PointFPModule(mlp_channels=cur_fp_mlps))
            if fp_index != len(fp_channels) - 1:
                fp_source_channel = cur_fp_mlps[-1]
                fp_target_channel = skip_channel_list.pop()

        self.GCN_Blocks = nn.ModuleList()
        for i in range(3):
            self.GCN_Blocks.append( GCN_Block(gcn_cfg[i]) )

    @auto_fp16(apply_to=('points', ))
    def forward(self, points):
        """Forward pass.

        Args:
            points (torch.Tensor): point coordinates with features,
                with shape (B, N, 3 + input_feature_dim).

        Returns:
            dict[str, list[torch.Tensor]]: Outputs after SA and FP modules.

                - fp_xyz (list[torch.Tensor]): The coordinates of \
                    each fp features.
                - fp_features (list[torch.Tensor]): The features \
                    from each Feature Propagate Layers.
                - fp_indices (list[torch.Tensor]): Indices of the \
                    input points.
        """
        xyz, features = self._split_point_feats(points)

        batch, num_points = xyz.shape[:2]
        indices = xyz.new_tensor(range(num_points)).unsqueeze(0).repeat(
            batch, 1).long()

        sa_xyz = [xyz]
        sa_features = [features]
        sa_indices = [indices]

        for i in range(self.num_sa):
            cur_xyz, cur_features, cur_indices = self.SA_modules[i](
                sa_xyz[i], sa_features[i])

            if i>0: 
                edges = build_edge(cur_xyz, k=32)
                edges = edges.permute(0, 2,1) # b x e x 2

                cur_features = self.GCN_Blocks[i-1](cur_xyz, cur_features, edges)

            sa_xyz.append(cur_xyz)
            sa_features.append(cur_features)
            sa_indices.append(
                torch.gather(sa_indices[-1], 1, cur_indices.long()))

        fp_xyz = [sa_xyz[-1]]
        fp_features = [sa_features[-1]]
        fp_indices = [sa_indices[-1]]

        for i in range(self.num_fp):
            fp_features.append(self.FP_modules[i](
                sa_xyz[self.num_sa - i - 1], sa_xyz[self.num_sa - i],
                sa_features[self.num_sa - i - 1], fp_features[-1]))
            fp_xyz.append(sa_xyz[self.num_sa - i - 1])
            fp_indices.append(sa_indices[self.num_sa - i - 1])


        ret = dict(
            fp_xyz=fp_xyz, fp_features=fp_features, fp_indices=fp_indices)
        return ret
