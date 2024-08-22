
from __future__ import annotations

import torch
import torch.nn as nn

from collections.abc import Sequence

import pdb


from InstanSeg.utils.utils import show_images

from einops import rearrange, repeat




from InstanSeg.utils.models.InstanSeg_UNet import DecoderBlock, EncoderBlock, conv_norm_act

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


class ChannelInvariantNet_1(nn.Module):
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
    



class ChannelInvariantNet_2(nn.Module):
    """
    ChannelInvariantNet:
    This is a model that implements a channel invariant adaptor network that takes an B,C,H,W input and returns a B,C_out,H,W output.
    """

    def __init__(self, 
            out_channels = 3,
            bias: bool = True,
            dropout: float | tuple = 0.0,
            upsample: str = "nontrainable",
            interp_mode: str = "linear",
            features: Sequence[int] = (8, 16, 32),
            act: str | tuple = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
            norm: str | tuple = ("instance", {"affine": True}),):
        
        from monai.networks.layers.factories import Conv
        from monai.utils import ensure_tuple_rep
        from monai.networks.nets.basic_unet import UpCat, TwoConv, Down, Conv


        super().__init__()

        fea = ensure_tuple_rep(features, 3)

        self.conv_init = TwoConv(2, 1, fea[0], act, norm, bias, dropout)

        self.conv_0 = TwoConv(2, fea[0]*2, fea[0], act, norm, bias, dropout)
        self.down_1 = Down(2, fea[0], fea[1], act, norm, bias, dropout)
        
        self.conv_1 = TwoConv(2, fea[1]*2, fea[1], act, norm, bias, dropout)
        self.down_2 = Down(2, fea[1], fea[2], act, norm, bias, dropout)

        self.conv_u2 = TwoConv(2, fea[2]*2, fea[2], act, norm, bias, dropout)
        self.upcat_2 = UpCat(2, fea[2], fea[1], fea[1], act, norm, bias, dropout, upsample, interp_mode = interp_mode)

        self.conv_u1 = TwoConv(2, fea[1]*2, fea[1], act, norm, bias, dropout)
        self.upcat_1 = UpCat(2, fea[1], fea[0] , fea[0], act, norm, bias, dropout, upsample, interp_mode = interp_mode)

        self.final_conv = Conv["conv", 2](fea[0], out_channels, kernel_size=1)

    def forward(self,x):

        #Example x is torch.Size([2, 4, 256, 256]) b,c,H,W
        b,c,h,w = x.shape
        x = rearrange(x, 'b c h w -> (b c) 1 h w') # Input size: (2, 4, 256, 256), Output size: torch.Size([8, 1, 256, 256])
        x0 = self.conv_init(x) # Output size: torch.Size([8, 8, 256, 256])
        x0 = rearrange(x0, '(b c) k h w -> b c k h w', b=b) # Output size: torch.Size([2, 4, 8, 256, 256])
        pool_x0 = x0.max(1)[0] # Output size: torch.Size([2, 8, 256, 256])
        cat0 = torch.cat((repeat(pool_x0, 'b k h w -> b c k h w', c=c), x0), dim=2) # Output size: torch.Size([2, 4, 16, 256, 256])
        cat0 = rearrange(cat0, 'b c k h w -> (b c) k h w') # Output size: torch.Size([8, 16, 256, 256])
        cat0 = self.conv_0(cat0) # Output size: torch.Size([8, 8, 256, 256])

        x1 = self.down_1(cat0) # Output size: torch.Size([8, 16, 128, 128])
        x1 = rearrange(x1, '(b c) k h w -> b c k h w', b=b) # Output size: torch.Size([2, 4, 16, 128, 128])
        pool_x1 = x1.max(1)[0] # Output size: torch.Size([2, 8, 128, 128])
        cat1 = torch.cat((repeat(pool_x1, 'b k h w -> b c k h w', c=c), x1), dim=2) # Output size: torch.Size([2, 4, 32, 128, 128])
        cat1 = rearrange(cat1, 'b c k h w -> (b c) k h w') # Output size: torch.Size([8, 32, 128, 128])
        cat1 = self.conv_1(cat1) # Output size: torch.Size([8, 16, 128, 128])

        x2 = self.down_2(cat1) # Output size: torch.Size([8, 32, 64, 64])
        x2 = rearrange(x2, '(b c) k h w -> b c k h w', b=b) # Output size: torch.Size([2, 4, 32, 64, 64])
        pool_x2 = x2.max(1)[0] # Output size: torch.Size([2, 32, 64, 64])
        cat2 = torch.cat((repeat(pool_x2, 'b k h w -> b c k h w', c=c), x2), dim=2) # Output size: torch.Size([2, 4, 64, 64, 64])
        cat2 = rearrange(cat2, 'b c k h w -> (b c) k h w') # Output size: torch.Size([8, 64, 64, 64])
        cat2 = self.conv_u2(cat2) # Output size: torch.Size([8, 32, 64, 64])

        u2 = self.upcat_2(cat2, cat1) # Output size: torch.Size([8, 16, 128, 128])
        u2 = rearrange(u2, '(b c) k h w -> b c k h w', b=b) # Output size: torch.Size([2, 4, 16, 128, 128])
        pool_u2 = u2.max(1)[0] # Output size: torch.Size([2, 16, 128, 128])
        cat_u2 = torch.cat((repeat(pool_u2, 'b k h w -> b c k h w', c=c), u2), dim=2) # Output size: torch.Size([2, 4, 32, 128, 128])
        cat_u2 = rearrange(cat_u2, 'b c k h w -> (b c) k h w') # Output size: torch.Size([8, 32, 128, 128])
        cat_u2 = self.conv_u1(cat_u2) # Output size: torch.Size([8, 16, 128, 128])

        u1 = self.upcat_1(cat_u2, cat0) # Output size: torch.Size([8, 8, 256, 256])
        u1 = rearrange(u1, '(b c) k h w -> b c k h w', b=b) # Output size: torch.Size([2, 4, 8, 256, 256])
        pool_u1 = u1.max(1)[0] # Output size: torch.Size([2, 8, 256, 256])

        u0 = self.final_conv(pool_u1) # Output size: torch.Size([2, 3, 256, 256])

        return u0
    



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
                model.AdaptorNet=ChannelInvariantNet_1(**kwargs)
            elif adaptor_net_str == "1_ablated":
                model.AdaptorNet=ChannelInvariantNet_1(aggregation= "no_aggregation",**kwargs)
            elif adaptor_net_str == "2":
                model.AdaptorNet=ChannelInvariantNet_2(**kwargs)
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

    adaptor_model = ChannelInvariantNet_1(out_channels=3)

    output  = adaptor_model(sample_input)

    print("Input size:", input_size)
    print("Compressed size:", output.size())
