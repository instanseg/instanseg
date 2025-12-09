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
    All parameters are trainable by default.

    Installation instructions:
    pip install natten==0.21.1+torch290cu130 -f https://whl.natten.org
    """
    
    def __init__(self, nout: int = 5, ps: int = 8, bsize: int = 256, 
                 checkpoint_path: str = "~/.sam/sam_vit_l_0b3195.pth"):
        super().__init__()
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Fixed internal feature dimension for SAM encoder
        self.encoder_dim = 256
        
        # Load SAM via cellpose Transformer (supports variable input sizes)
        self.image_encoder = Transformer(
            nout=self.encoder_dim, 
            ps=ps, 
            bsize=bsize, 
            checkpoint=os.path.expanduser(checkpoint_path)
        )
        
        # Load NAF for upsampling features to input resolution
        self.naf = torch.hub.load("valeoai/NAF", "naf", pretrained=True, device=device)
        
        # 1x1 conv to project from encoder_dim to desired nout
        self.proj = nn.Conv2d(self.encoder_dim, nout, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the SAM image encoder with NAF upsampling.
        
        Args:
            x: Input tensor of shape [B, 3, H, W]. 
               Input size should match bsize (default 256x256).
               
        Returns:
            Feature map of shape [B, nout, H, W] (same spatial dims as input)
        """
        # Get low-resolution features from SAM encoder [B, encoder_dim, H/ps, W/ps]
        lr_features = self.image_encoder(x)[0]
        
        # Upsample to input resolution using NAF [B, encoder_dim, H, W]
        target_size = (x.shape[2], x.shape[3])  # (H, W)
        upsampled = self.naf(x, lr_features, target_size)
        
        # Project to desired output channels [B, nout, H, W]
        return self.proj(upsampled)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SAMFeatureExtractor(nout=5).to(device)
    
    x = torch.randn(1, 3, 256, 256).to(device)
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

