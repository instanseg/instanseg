
from __future__ import annotations
import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F



def conv_norm_act(in_channels, out_channels, sz,norm, act = "ReLU", depthwise = False):

    if norm == "None" or norm is None:
        norm_layer = nn.Identity()
    elif norm.lower() == "batch":
        norm_layer = nn.BatchNorm2d(out_channels,eps = 1e-5, momentum = 0.05)
    elif norm.lower() == "instance":
        norm_layer = nn.InstanceNorm2d(out_channels,eps = 1e-5, track_running_stats=False, affine = True)
    else:
        raise ValueError("Norm must be None, batch or instance")
    
    if act == "None" or act is None:
        act_layer = nn.Identity()
    elif act.lower() == "relu":
        act_layer = nn.ReLU(inplace=True)
    elif act.lower() == "relu6":
        act_layer = nn.ReLU6(inplace=True)
    elif act.lower() == "mish":
        act_layer = nn.Mish(inplace=True)
    else:
        raise ValueError("Act must be None, ReLU or Mish")
    
    if depthwise:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, sz, padding=sz//2, groups = in_channels),
            norm_layer,
            act_layer,
        )
    else:
        
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, sz, padding=sz//2),
            norm_layer,
            act_layer,
        )  


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels,
            norm = "BATCH",
            act = "ReLU",
            shallow = False,
    ):
        super().__init__()

        
        self.conv0 = conv_norm_act(in_channels, out_channels, 1,norm,act)
        self.conv_skip = conv_norm_act(skip_channels, out_channels, 1,norm,act)
        self.conv1 = conv_norm_act(in_channels, out_channels, 3,norm,act)
        self.conv2 = conv_norm_act(out_channels, out_channels, 3,norm,act)
        self.conv3 = conv_norm_act(out_channels, out_channels, 3,norm,act)
        self.conv4 = conv_norm_act(out_channels, out_channels, 3,norm,act)

        if shallow:
            self.conv3 = nn.Identity()

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        proj = self.conv0(x)
        x = self.conv1(x)
        x =  proj + self.conv2(x + self.conv_skip(skip))
        x = x + self.conv4(self.conv3(x))
        return x


class EncoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            pool = True,
            norm = "BATCH",
            act = "ReLU",
            shallow = False,
    ):
        super().__init__()

        if pool:
            self.maxpool = nn.MaxPool2d(2, 2)
        else:
            self.maxpool = nn.Identity()
        self.conv0 = conv_norm_act(in_channels, out_channels, 1,norm,act)
        self.conv1 = conv_norm_act(in_channels, out_channels, 3,norm,act)
        self.conv2 = conv_norm_act(out_channels, out_channels, 3,norm,act)
        self.conv3 = conv_norm_act(out_channels, out_channels, 3,norm,act)
        self.conv4 = conv_norm_act(out_channels, out_channels, 3,norm,act)

        if shallow:
            self.conv2 = nn.Identity()
            self.conv3 = nn.Identity()

    def forward(self, x):

        x = self.maxpool(x)
        proj = self.conv0(x)
        x = self.conv1(x)
        x =  proj + self.conv2(x)
        x = x + self.conv4(self.conv3(x))
        return x
    
class Decoder(nn.Module):
    def __init__(
            self,
            layers,
            out_channels,
            norm,
            act):
        super().__init__()
        
        self.decoder = nn.ModuleList([DecoderBlock(layers[i],layers[i+1],layers[i+1],norm = norm, act =  act) for i in range(len(layers)-1)])
        
        self.final_block = nn.ModuleList([conv_norm_act(layers[-1],out_channel,1, norm = norm if (norm is not None) and norm.lower() != "instance" else None,act = None) for out_channel in out_channels])
        
    def forward(self,x,skips):
        for layer,skip in zip(self.decoder,skips[::-1]):
            x = layer(x,skip)

        x = torch.cat([final_block(x) for final_block in self.final_block],dim = 1)
        return x
            
    
class InstanSeg_UNet(nn.Module):
    def __init__(self,in_channels,out_channels,layers = [256,128,64,32],norm = "BATCH",dropout = 0, act = "ReLu"):
        super().__init__()
        layers = layers[::-1]
        self.encoder = nn.ModuleList([EncoderBlock(in_channels,layers[0],pool = False,norm = norm, act = act)] + [EncoderBlock(layers[i],layers[i+1],norm = norm, act = act) for i in range(len(layers)-1)])
        layers = layers[::-1]

        # out_channels should be a list of lists [[2,2,1],[2,2,1]] means two decoders, each with 3 output blocks. The output will be of shape 10.

        if type(out_channels) == int:
            out_channels = [[out_channels]]
        if type(out_channels[0]) == int:
            out_channels = [out_channels]
            
        self.decoders = nn.ModuleList([Decoder(layers,out_channel,norm, act) for out_channel in out_channels])
    
    def forward(self,x):
        skips = []
        for n,layer in enumerate(self.encoder):
            x = layer(x)
            if n < len(self.encoder)-1:
                skips.append(x)

        return torch.cat([decoder(x,skips) for decoder in self.decoders],dim = 1)
    



if __name__ == "__main__":

    net = InstanSeg_UNet(
        in_channels=3,
        out_channels=[5,3],
    )

    print(net(torch.randn(1,3,256,256)).shape)
