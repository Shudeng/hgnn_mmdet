import torch
from torch import nn
import numpy as np
from utils import multi_layer_neural_network_fn
from torch.nn.functional import gumbel_softmax


EPSILON = np.finfo(np.float32).tiny

def gumbel_keys(w):
    uniform = w.new(w.shape).uniform_(EPSILON, 1)
#    return uniform + torch.sigmoid(w)
    z = -torch.log(-torch.log(uniform))
    return torch.log(w+1e-8) + z
  

def continuous_topk(w, k, t=0.1, separate=True, hard=True):
    khot_list = []
    w = w.unsqueeze(1)
    onehot_approx = torch.zeros_like(w, dtype=torch.float32)

    def to_onehot(w):
        b, out_num, in_num = w.shape
        argmax = torch.argmax(w, dim=-1).view(-1) # b x out_num
        onehot = torch.zeros((argmax.shape[0], in_num), device=argmax.device)
        onehot[torch.arange(onehot.shape[0]), argmax.long()] = 1
        onehot = onehot.view(b, out_num, in_num)
        return onehot

    for i in range(k):
        #khot_mask = torch.maximum(1.0 - onehot_approx, EPSILON)
        khot_mask = torch.clamp(1.0 - onehot_approx, min=EPSILON)
        w += torch.log(khot_mask)
        onehot_approx = torch.nn.functional.softmax(w / t, dim=-1)
        #print("onehot_approx.max(-1)", onehot_approx.max(-1))
        #print("onehot_approx.argmax(-1)", onehot_approx.argmax(-1))
        if hard:
            onehot_approx = to_onehot(onehot_approx) - onehot_approx.detach() + onehot_approx
        khot_list.append(onehot_approx)

    khot_list = torch.cat(khot_list, dim=1)
    if separate:
        return khot_list
    else:
        return khot_list.sum(0)

class DownSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, in_num_points=2048, out_num_points=1024):
        super(DownSampleBlock, self).__init__()
        #from utils import MLPs
        #self.mlps = MLPs([int(in_channels), int(out_channels), int(out_channels)])
        #self.downsample_mlp = MLPs([int(out_channels), int(out_channels)//2, 1])
        
        self.mlps = multi_layer_neural_network_fn([int(in_channels), int(out_channels), int(out_channels)], norm="LayerNorm")
        self.downsample_mlp = multi_layer_neural_network_fn([int(out_channels), int(out_channels)//2, 1], relu=True, norm="LayerNorm")
        self.out_num_points = int(out_num_points)
        
    def regulizer(self, w):
        # avoid select the same point
        # w.shape: b x out_num x in_num
        reg_loss = torch.clamp(w.sum(dim=1)-1, min=0).mean()
        
        return reg_loss
        
        
    def forward(self, xyzs, features):
        # feature: b x c x num_points
        features = features.permute(0, 2, 1)
        features = self.mlps(features)


        w = self.downsample_mlp(features) # b x in_num x 1

        if self.training:
            w = gumbel_keys(w)

        w_prime = continuous_topk(w[:,:, 0], self.out_num_points, t=0.1, hard=True) # b x out_num x in_num
        #onehot = torch.nn.functional.onehot(onehot.view(-1), num_classes=in_num)
        #w_prime = onehot

        #w_prime = (onehot - w.detach) + w
        #w_prime = gumbel_softmax(w, tau=1, hard=True, dim=1) # b x in_num x out_num
        #w_prime = w_prime.permute(0, 2, 1) # b x out_num x in_num

        xyzs = w_prime @ xyzs
        features = w_prime @ features
        features = features.permute(0, 2, 1)
        
        indices = w_prime.argmax(dim=2)
        
        self.loss = self.regulizer(w_prime)
        return xyzs, features, indices
        
if __name__=="__main__":
    print("EPSILON", EPSILON)
    w = torch.rand((1, 256))
    w = gumbel_keys(w)
    w = continuous_topk(w, 128, 1)
    print("w.argmax(-1)", w.argmax(-1))

    w_list = w.argmax(-1).tolist()
    for i in range(len(w_list)):
        print(i, "len(w[0])", len(set(w[i])))


    exit(0)
    
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
    
    
    
    
    
