import segmentation_models_pytorch as smp
from torch import nn


class SegFormer(nn.Module):
    def __init__(self, in_channels,out_channels, encoder = "mit_b5", decoder = "MAnet",weights= None):
        super().__init__()

        net_shape = smp.MAnet if decoder == "MAnet" else smp.FPN

        self.net = net_shape(
            encoder_name=encoder,
            encoder_weights=weights,
            in_channels=in_channels,
            classes=out_channels,
            activation=None)

    def forward(self, x):

        return self.net(x)


