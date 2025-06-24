import torch
import torch.nn as nn
from cellpose.vit_sam import Transformer
import os


class CellposeSam(nn.Module):
    def __init__(self, nout=5):
        super(CellposeSam, self).__init__()

        self.model = Transformer(nout = nout, checkpoint= os.path.expanduser("~/.sam/sam_vit_l_0b3195.pth"))


    def forward(self, x):
        return self.model(x)[0]

