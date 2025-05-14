import torch
import torch.nn as nn
from cellpose.vit_sam import Transformer
from cellpose import models

class CellposeSam(nn.Module):
    def __init__(self, nout=5, gpu=True):
        super(CellposeSam, self).__init__()
        # Initialize the Transformer model
        self.model = Transformer(nout=3)
        self.gpu = gpu
        
        # Load the Cellpose pretrained weights if available
        self.load_pretrained_weights()

        self.set_output_channels(nout)
        
    def load_pretrained_weights(self):
        try:
            # Load the pretrained Cellpose model weights
            cellpose_model = models.CellposeModel(gpu=self.gpu)
            cellpose_state_dict = cellpose_model.net.state_dict()
            
            # Check for compatibility and load weights
            model_state_dict = self.model.state_dict()
            model_state_dict.update(cellpose_state_dict)
            self.model.load_state_dict(model_state_dict)
            print("Pretrained weights loaded successfully.")
        except Exception as e:
            print(f"Error loading pretrained weights: {e}")
    
    def set_output_channels(self, out_channels):

        in_channels = self.model.out.in_channels
        kernel_size = self.model.out.kernel_size
        stride = self.model.out.stride
        padding = self.model.out.padding
        
        # Replace the output layer
        self.model.out = nn.Conv2d(
            in_channels, out_channels * self.model.ps **2, kernel_size=kernel_size, stride=stride, padding=padding
        )
        # Reinitialize the new layer
        nn.init.kaiming_normal_(self.model.out.weight, mode='fan_out', nonlinearity='relu')
        if self.model.out.bias is not None:
            nn.init.constant_(self.model.out.bias, 0)
        print(f"Output layer updated to {out_channels} channels.")

        self.model.W2 = nn.Parameter(torch.eye(out_channels* self.model.ps**2).reshape(out_channels*self.model.ps**2, out_channels, self.model.ps, self.model.ps), 
                        requires_grad=False)


    def forward(self, x):
        return self.model(x)[0]

