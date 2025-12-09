import os
import torch
import torch.nn as nn
from cellpose.vit_sam import Transformer


class SAMFeatureExtractor(nn.Module):
    """
    SAM (Segment Anything Model) wrapper that outputs feature maps at the same 
    spatial resolution as the input, using NAF for upsampling.
    
    Uses cellpose's Transformer which supports variable input sizes via bsize parameter.
    NAF upsamples features back to [B, C, H, W].
    
    Only the input_conv and proj layers are trainable; SAM and NAF are frozen.

    Installation instructions:
    pip install natten==0.21.1+torch290cu130 -f https://whl.natten.org
    """
    
    def __init__(self, nout: int = 5, ps: int = 8, bsize: int = 256, 
                 checkpoint_path: str = "~/.sam/sam_vit_l_0b3195.pth",
                 in_channels: int = 3):
        super().__init__()
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Fixed internal feature dimension for SAM encoder
        self.encoder_dim = 256
        
        # Learnable input conv block to normalize/preprocess data for SAM
        # Input: [B, in_channels, H, W] -> Output: [B, 3, H, W]
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        
        # Load SAM via cellpose Transformer (supports variable input sizes)
        self.image_encoder = Transformer(
            nout=self.encoder_dim, 
            ps=ps, 
            bsize=bsize, 
            checkpoint=os.path.expanduser(checkpoint_path)
        )
        
        # Load NAF for upsampling features to input resolution
        self.naf = torch.hub.load("valeoai/NAF", "naf", pretrained=True, device=device)
        
        # Freeze pretrained SAM and NAF layers
        # for param in self.image_encoder.parameters():
        #     param.requires_grad = False
        # for param in self.naf.parameters():
        #     param.requires_grad = False
        
        # 1x1 conv to project from encoder_dim to desired nout (trainable)
        self.proj = nn.Conv2d(self.encoder_dim, nout, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the SAM image encoder with NAF upsampling.
        
        Args:
            x: Input tensor of shape [B, in_channels, H, W]. 
               Input size should match bsize (default 256x256).
               
        Returns:
            Feature map of shape [B, nout, H, W] (same spatial dims as input)
        """
        # Learnable normalization/preprocessing [B, in_channels, H, W] -> [B, 3, H, W]
        x_norm = self.input_conv(x)
        
        # Get low-resolution features from SAM encoder [B, encoder_dim, H/ps, W/ps]
        lr_features = self.image_encoder(x_norm)[0]
        
        # Upsample to input resolution using NAF [B, encoder_dim, H, W]
        target_size = (x.shape[2], x.shape[3])  # (H, W)
        upsampled = self.naf(x_norm, lr_features, target_size)
        
        # Project to desired output channels [B, nout, H, W]
        return self.proj(upsampled)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SAMFeatureExtractor(nout=5, in_channels=3).to(device)
    
    x = torch.randn(1, 3, 256, 256).to(device)
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Print trainable vs frozen parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")