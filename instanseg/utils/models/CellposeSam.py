import torch
import torch.nn as nn
from cellpose.vit_sam import Transformer
import os


class CellposeSam(nn.Module):
    def __init__(self, nout=5, ps = 8, bsize = 256):
        super(CellposeSam, self).__init__()

        self.model = Transformer(nout = nout, ps = ps,bsize= bsize,checkpoint= os.path.expanduser("~/.sam/sam_vit_l_0b3195.pth"))


    def forward(self, x):
        return self.model(x)[0]



from instanseg.utils.models.InstanSeg_UNet import EncoderBlock, Decoder
from torch import nn

class SAM_UNet(nn.Module):
    def __init__(self, in_channels, out_channels, layers=[64,32], norm="BATCH", dropout=0, act="ReLu"):
        super().__init__()

        print(layers)
        layers = layers[::-1]

        self.encoder = nn.ModuleList(
            [EncoderBlock(in_channels, layers[0], pool=False, norm=norm, act=act)] +
            [EncoderBlock(layers[i], layers[i+1], norm=norm, act=act) for i in range(len(layers)-1)]
        )

        


        bsize = 128 if len(layers) == 2 else 64 

        print(f"Using Cellpose SAM with bsize: {bsize} and layers: {layers}")
        model = CellposeSam(nout= layers[-1], bsize = bsize).to("cuda")
        model.model.encoder.patch_embed.proj = torch.nn.Conv2d(layers[-1], 1024, kernel_size=(8, 8), stride=(8, 8))

        self.vit_model = model

        layers = layers[::-1]

        if type(out_channels) == int:
            out_channels = [[out_channels]]
        if type(out_channels[0]) == int:
            out_channels = [out_channels]

        self.decoders = nn.ModuleList([
            Decoder(layers, out_channel, norm, act) for out_channel in out_channels
        ])

    def forward(self, x):
        skips = []
        for n, layer in enumerate(self.encoder):
            x = layer(x)
            if n < len(self.encoder) - 1:
                skips.append(x)

        print(f"Shape of bottleneck feature map: {x.shape}")
            
        x = self.vit_model(x)  # <-- Pass bottleneck feature map through ViT

        print(f"Shape after ViT: {x.shape}")

        return torch.cat([decoder(x, skips) for decoder in self.decoders], dim=1)
