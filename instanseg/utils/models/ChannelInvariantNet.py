
from __future__ import annotations

import torch
import torch.nn as nn

from collections.abc import Sequence

import pdb


from instanseg.utils.utils import show_images

from einops import rearrange, repeat




from instanseg.utils.models.InstanSeg_UNet import DecoderBlock, EncoderBlock, conv_norm_act

class ChannelInvariantEncoderBlock(EncoderBlock):
    def __init__(
            self,
            in_channels,
            out_channels,
            norm = "BATCH",
            act = "ReLU",
            aggregation = "sum",
            pool = True,
    ):
        super().__init__(in_channels, out_channels,norm =  norm, act = act, pool = pool, shallow=False)
        self.aggregation = aggregation

        if self.aggregation in ["concat", "no_aggregation"]:
            self.reduce_block = conv_norm_act(out_channels * 2, out_channels, sz = 3, norm = norm, act = act)

    def forward(self, x, c, b):

        
        x0 = super().forward(x)
        x0 = rearrange(x0, '(b c) k h w -> b c k h w', b = b)
        pool_x0 = x0.max(1)[0] # b k h w

        if self.aggregation == "concat":
            cat0 = torch.cat((repeat(pool_x0, 'b k h w -> b c k h w', c = c), x0), dim = 2)
            cat0 = rearrange(cat0, 'b c k h w -> (b c) k h w')
            cat0 = self.reduce_block(cat0)
        if self.aggregation == "no_aggregation":
            cat0 = torch.cat((repeat(pool_x0, 'b k h w -> b c k h w', c = c) * 0, x0), dim = 2)
            cat0 = rearrange(cat0, 'b c k h w -> (b c) k h w')
            cat0 = self.reduce_block(cat0)

        elif self.aggregation == "sum":
            cat0 = repeat(pool_x0, 'b k h w -> b c k h w', c = c) + x0
            cat0 = rearrange(cat0, 'b c k h w -> (b c) k h w')

        return cat0


class ChannelInvariantDecoderBlock(DecoderBlock):
    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels,
            aggregation = "sum",
            final_decoder = False,
            norm = "BATCH",
            act = "ReLU",
    ):
        super().__init__(in_channels, skip_channels, out_channels, norm, act, shallow = False)
        self.aggregation = aggregation
        self.final_decoder = final_decoder

        if self.aggregation in ["concat", "no_aggregation"]:
            self.reduce_block = conv_norm_act(out_channels * 2, out_channels, sz = 3, norm = norm, act = act)

    def forward(self, x, skip=None, c = None, b = None):

        u2 = super().forward(x, skip)
        u2 = rearrange(u2, '(b c) k h w -> b c k h w', b = b)
        pool_u2 = u2.max(1)[0] # b k h w

        if self.final_decoder:
            return pool_u2

        if self.aggregation == "concat":
            cat_u2 = torch.cat((repeat(pool_u2, 'b k h w -> b c k h w', c = c), u2), dim = 2)
            cat_u2 = rearrange(cat_u2, 'b c k h w -> (b c) k h w')
            cat_u2 = self.reduce_block(cat_u2)

        if self.aggregation == "no_aggregation":
            cat_u2 = torch.cat((repeat(pool_u2, 'b k h w -> b c k h w', c = c) * 0, u2), dim = 2)
            cat_u2 = rearrange(cat_u2, 'b c k h w -> (b c) k h w')
            cat_u2 = self.reduce_block(cat_u2)

        elif self.aggregation == "sum":
            cat_u2 = repeat(pool_u2, 'b k h w -> b c k h w', c = c) + u2
            cat_u2 = rearrange(cat_u2, 'b c k h w -> (b c) k h w')

        
        return cat_u2


class ChannelInvariantNet(nn.Module):
    """
    ChannelInvariantNet:
    This is a model that implements a channel invariant adaptor network that takes an B,C,H,W input and returns a B,C_out,H,W output.
    """

    def __init__(self, 
            out_channels = 3,
            layers = [32,16,8],
            act = "ReLu",
            norm = "BATCH",
            aggregation = "concat",): #sum or concat

        super().__init__()

        layers = layers[::-1]

        self.aggregation = aggregation

        self.conv_0 = ChannelInvariantEncoderBlock(1, layers[0], act = act, norm  = norm, pool = False, aggregation  = aggregation)

        self.down_1 = ChannelInvariantEncoderBlock(layers[0] , layers[1], act = act, norm = norm, pool = True, aggregation  = aggregation)
        self.down_2 = ChannelInvariantEncoderBlock(layers[1] , layers[2], act = act, norm = norm, pool = True, aggregation  = aggregation)

        self.upcat_2 = ChannelInvariantDecoderBlock(layers[2] , layers[1] , layers[1] , act = act, norm = norm, aggregation  = aggregation)
        self.upcat_1 = ChannelInvariantDecoderBlock(layers[1] , layers[0] , layers[0] ,act =  act,norm =  norm, final_decoder=True, aggregation  = aggregation)

        final_norm = norm if (norm is not None) and norm.lower() != "instance" else None

        self.final_conv = conv_norm_act(layers[0] ,out_channels, sz = 1, norm = final_norm,act = None)

    def forward(self,x):

        b,c = x.shape[:2]

        x = rearrange(x, 'b c h w -> (b c) 1 h w')
    
        cat0 = self.conv_0(x, c = c, b = b)

        cat1 = self.down_1(cat0, c = c, b = b)

        cat2 = self.down_2(cat1, c = c, b = b)

        cat_u2 = self.upcat_2(cat2, cat1, c = c, b = b)
    
        pool_u1 = self.upcat_1(cat_u2, cat0, c = c, b = b)

        u0 = self.final_conv(pool_u1)

        return u0.float()
    



def has_AdaptorNet(model):
    for module in model.modules():
        if isinstance(module, torch.nn.Module):
            module_class = module.__class__.__name__
            if 'AdaptorNet' in module_class:
                return True
    return False

def initialize_AdaptorNet(model,adaptornet=None, adaptor_net_str = "1",**kwargs):
    if has_AdaptorNet(model):
        return model
    else:
        if adaptornet is not None:
            model.AdaptorNet=adaptornet
            return model
        else:
            if adaptor_net_str == "1":
                model.AdaptorNet=ChannelInvariantNet(**kwargs)
            elif adaptor_net_str == "1_ablated":
                model.AdaptorNet=ChannelInvariantNet(aggregation= "no_aggregation",**kwargs)
        return model
    
class _AdaptorNetWrapper(torch.nn.Module):
    def __init__(self,model,adaptornet=None,adaptor_net_str = "1", **kwargs):
        super().__init__()

        self.model=initialize_AdaptorNet(model,adaptornet=adaptornet,adaptor_net_str = adaptor_net_str, **kwargs)

    def forward(self,x):

        x = self.model.AdaptorNet(x)
        out = self.model(x)
        return out
    

class AdaptorNetWrapper(_AdaptorNetWrapper):
    def __init__(self, *args, device='cuda', norm = None, **kwargs):

        if norm is not None and norm.lower() in ["instance_invariant", "custom_instance_invariant"]:
            _norm = "INSTANCE"
        else:
            _norm = norm
        super().__init__(norm = _norm,*args, **kwargs)

        # self.set_running_stats(device)
            
    def set_running_stats(self, device, model=None):
        if not model:
            model = self
        for child_name, child in model.named_children():
            if isinstance(child, torch.nn.InstanceNorm2d):
                child.track_running_stats = True
                child.register_buffer('running_mean', torch.zeros(child.num_features, device=device))
                child.register_buffer('running_var', torch.ones(child.num_features, device=device))
                child.register_buffer('num_batches_tracked', torch.ones(1, device=device))
            else:
                self.set_running_stats(device, child)  
    



if __name__ == "__main__":

    import torch
    import matplotlib.pyplot as plt
    import torch.nn.functional as F


    import pdb

    #  B batch size, C channels, H height, W width
    input_size = (2, 4, 256, 256)
    sample_input = torch.randn(input_size)

    adaptor_model = ChannelInvariantNet(out_channels=3)

    output  = adaptor_model(sample_input)

    print("Input size:", input_size)
    print("Compressed size:", output.size())
