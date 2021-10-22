import torch
import math

#from torch_scatter import scatter_max, scatter
#from torch_cluster import radius, radius_graph
#from torch_cluster import knn_graph

from mmcv.runner import auto_fp16
from mmcv import Config
from torch import nn as nn
from mmdet3d.ops import PointFPModule, build_sa_module
from mmdet3d.models.backbones.base_pointnet import BasePointNet

from mmdet.models import BACKBONES
from utils import build_edge, GCN_Block
from downsample_block import DownSampleBlock



class TransformerPoint(BasePointNet):
    def __init__(self,
                 in_channels,
                 num_points=(2048, 1024, 512, 256),
                 fps_layer_config=None,
                 gan_cfg = [3, 3, 3, 3],
                 fp_channels=((256, 256), (256, 256)),
                ):

        super().__init__()
        
        self.layers = len(num_points)
        self.num_fp = len(fp_channels)
        
        
        self.fps_layer = build_sa_module(
            num_point=num_points[0],
            radius=fps_layer_config.radius,
            num_sample=fps_layer_config.num_sample,
            mlp_channels=[in_channels-3] + list(fps_layer_config.mlp_channels),
            norm_cfg=fps_layer_config.norm_cfg,
            cfg=fps_layer_config.sa_cfg)
            
        self.transformer_blocks = nn.ModuleList()
        self.downsample_blocks = nn.ModuleList()
        for layer in range(self.layers):
            self.transformer_blocks.append( GCN_Block(gan_cfg[layer], edge_MLP_depth_list=[128*2**layer+3, 128*2**layer], update_MLP_depth_list=[128*2**layer, 128*2**layer]) )
            
            if layer<self.layers-1:
                self.downsample_blocks.append( DownSampleBlock(in_channels=128*2**layer, out_channels=128*2**(layer+1), in_num_points=2048/(2**layer), out_num_points=2048/(2**(layer+1))) )
                
        self.FP_modules = nn.ModuleList()
        source_channels = 128*2**(self.layers-1)
        for layer in range(len(fp_channels)):
            target_channels = 128*2**(2-layer)
            cur_fp_mlps = [source_channels + target_channels] + list(fp_channels[layer])
            self.FP_modules.append( PointFPModule(mlp_channels=cur_fp_mlps) )
            source_channels = list(fp_channels)[layer][-1]

    def forward(self, points):
        xyz, features = self._split_point_feats(points)
        batch, num_points = xyz.shape[:2]
        indices = xyz.new_tensor(range(num_points)).unsqueeze(0).repeat(
                        batch, 1).long()
                        
        cur_xyz, cur_features, cur_indices = self.fps_layer(xyz, features)
        sa_xyz = [cur_xyz]
        sa_indices = [cur_indices]
        sa_features = []
        self.regulizer_loss = 0
        
        
        for layer in range(self.layers):
            edges = build_edge(sa_xyz[-1]).permute(0, 2, 1)
            
            # filter edges
            #print("edges.shape", edges.shape)
            cur_features = self.transformer_blocks[layer](sa_xyz[-1], cur_features, edges)
            sa_features.append( cur_features )
            #print("cur_features.shape", cur_features.shape)
            #print("xyzs.shape", cur_xyz.shape)
            if layer<self.layers-1:
                cur_xyz, cur_features, cur_indices = self.downsample_blocks[layer](sa_xyz[-1], cur_features)
                
                sa_xyz.append( cur_xyz )
                sa_indices.append( torch.gather(sa_indices[-1].long(), 1, cur_indices.long()) )
                self.regulizer_loss += self.downsample_blocks[layer].loss
                
        #print("loss", self.regulizer_loss)
        
        # upsample
        fp_xyz = [sa_xyz[-1]]
        fp_features = [sa_features[-1]]
        fp_indices = [sa_indices[-1]]
        
        assert len(sa_xyz) == len(sa_features) == len(sa_indices)
        
        sa_len = len(sa_xyz)
        
        for i in range(self.num_fp):
            fp_features.append(self.FP_modules[i](
                sa_xyz[-(i+2)], sa_xyz[-(i+1)],
                sa_features[-(i+2) ], fp_features[-1]))
                
            fp_xyz.append(sa_xyz[-(i+2)])
            fp_indices.append(sa_indices[-(i+2)])
            
            #print("fp_features[-1].shape", fp_features[-1].shape)
            #print("fp_xyz[-1].shape", fp_xyz[-1].shape)
            #print("fp_indices[-1].shape", fp_indices[-1].shape)
            
        ret = dict(fp_xyz=fp_xyz, fp_features=fp_features, fp_indices=fp_indices)
            
        return ret

in_channels=4
fps_layer_config = Config.fromfile("backbone_config.py")

BACKBONE = TransformerPoint(
    in_channels=in_channels,
    fps_layer_config=fps_layer_config
)



if __name__=="__main__":
    device = "cuda"
    in_channels=4
    fps_layer_config = Config.fromfile("backbone_config.py")
    model = TransformerPoint(
        in_channels=in_channels,
        fps_layer_config=fps_layer_config
    )
    model.to(device)

    points = torch.rand((2, 20000, 4)).to(device)
    out = model(points)

