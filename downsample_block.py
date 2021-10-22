import torch
from torch import nn
from utils import multi_layer_neural_network_fn
from torch.nn.functional import gumbel_softmax

class DownSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, in_num_points=2048, out_num_points=1024):
        super(DownSampleBlock, self).__init__()
        
        self.mlps = multi_layer_neural_network_fn([int(in_channels), int(out_channels)])
        self.downsample_mlp = multi_layer_neural_network_fn([int(out_channels), int(out_num_points)], relu=False)
        
    def regulizer(self, w):
        # avoid select the same point
        # w.shape: b x out_num x in_num
        reg_loss = torch.clamp(w.sum(dim=1)-1, min=0).mean()
        
        return reg_loss
        
        
    def forward(self, xyzs, features):
        # feature: b x c x num_points
        features = features.permute(0, 2, 1)
        features = self.mlps(features)
        
        w = self.downsample_mlp(features) # b x in_num x out_num
        
        """
        b, in_num, out_num = w.shape
        onehot = torch.argmax(w, dim=1)
        onehot = torch.onehot(onehot.view(-1), num_classes=in_num)
        onehot = onehot.view(b, out_num, -1).permute(0,2,1)
        """
        w_prime = gumbel_softmax(w, tau=1, hard=True, dim=1) # b x in_num x out_num
        w_prime = w_prime.permute(0, 2, 1) # b x out_num x in_num
        
        xyzs = w_prime @ xyzs
        features = w_prime @ features
        features = features.permute(0, 2, 1)
        
        indices = w_prime.argmax(dim=2)
        
        self.loss = self.regulizer(w_prime)
        return xyzs, features, indices
        
if __name__=="__main__":
    device = "cuda"
    xyzs = torch.rand((1, 2048, 3)).to(device)
    features = torch.rand((1, 2048, 128)).to(device)
    block = DownSampleBlock(in_channels=128, out_channels=256, in_num_points=2048, out_num_points=1024)
    
    print("xyzs.shape", xyzs.shape)
    print("features.shape", features.shape)
    block.to(device)
    xyzs, features = block(xyzs, features)
    
    print("xyzs.shape", xyzs.shape)
    print("features.shape", features.shape)
    print("block.loss", block.loss)
    
    
    
    
    
