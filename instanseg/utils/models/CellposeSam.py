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

        x = self.vit_model(x)  # <-- Pass bottleneck feature map through ViT

        return torch.cat([decoder(x, skips) for decoder in self.decoders], dim=1)



from torch import nn

import torch
from torch import nn
from typing import List, Tuple

from typing import List, Tuple
import torch


@torch.jit.script
def _chops(img_shape: List[int], shape: Tuple[int, int], overlap: int = 0) -> Tuple[List[int], List[int]]:
    """
    TorchScript-compatible function to compute starting indices for sliding windows.
    img_shape: [C, H, W] or [H, W]
    shape: (window_h, window_w)
    overlap: overlap between tiles
    Returns: (h_index, v_index)
    """
    h = int(img_shape[-2])
    v = int(img_shape[-1])

    if h < shape[0] or v < shape[1]:
        return [0], [0]

    assert shape[0] > overlap and shape[1] > overlap, "Overlap must be smaller than window size"

    stride_h = shape[0] - overlap
    stride_v = shape[1] - overlap

    max_v = v - shape[1]
    max_h = h - shape[0]

    # Generate indices for vertical (width)
    v_index: List[int] = []
    i = 0
    while i * stride_v <= max_v:
        v_index.append(i * stride_v)
        i += 1
    if v_index[-1] != max_v:
        v_index.append(max_v)

    # Generate indices for horizontal (height)
    h_index: List[int] = []
    j = 0
    while j * stride_h <= max_h:
        h_index.append(j * stride_h)
        j += 1
    if h_index[-1] != max_h:
        h_index.append(max_h)

    return h_index, v_index



@torch.jit.script
def _tiles_from_chops(
    image: torch.Tensor,
    shape: Tuple[int, int],
    tuple_index: Tuple[List[int], List[int]]
) -> List[torch.Tensor]:
    """
    TorchScript-friendly function to extract sliding window tiles.
    
    image: [C, H, W] or [H, W]
    shape: (tile_h, tile_w)
    tuple_index: (h_index, v_index) from _chops
    """
    h_index, v_index = tuple_index

    # Ensure channel dimension exists
    if image.dim() == 2:
        image = image.unsqueeze(0)

    stride_h = shape[0]
    stride_v = shape[1]

    tile_list: List[torch.Tensor] = []

    for i in range(len(h_index)):
        window_i = h_index[i]
        for j in range(len(v_index)):
            window_j = v_index[j]
            current_window = image[..., window_i:window_i + stride_h, window_j:window_j + stride_v]
            tile_list.append(current_window)

    return tile_list


@torch.jit.script
def _stitch_mean(
    tiles: torch.Tensor,
    shape: Tuple[int, int],
    chop_list: Tuple[List[int], List[int]],
    final_shape: Tuple[int, int, int]
) -> torch.Tensor:
    """
    TorchScript-friendly stitching function that averages overlapping tiles using a Gaussian weight mask.
    """
    device = tiles[0].device
    canvas = torch.zeros(final_shape, dtype=torch.float32, device=device)
    canvas_weights = torch.zeros(final_shape, dtype=torch.float32, device=device)

    H, W = shape
    y = torch.linspace(-1.0, 1.0, steps=H, device=device)
    x = torch.linspace(-1.0, 1.0, steps=W, device=device)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    sigma = 0.5
    gaussian_mask = torch.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))

    # Expand mask to match tile shape if needed
    while gaussian_mask.dim() < tiles[0].dim():
        gaussian_mask = gaussian_mask.unsqueeze(0)

    h_index, v_index = chop_list

    for i in range(len(h_index)):
        window_i = h_index[i]
        for j in range(len(v_index)):
            window_j = v_index[j]

            idx = i * len(v_index) + j
            tile = tiles[idx]
            weighted_tile = tile * gaussian_mask

            canvas[..., window_i:window_i + H, window_j:window_j + W] += weighted_tile
            canvas_weights[..., window_i:window_i + H, window_j:window_j + W] += gaussian_mask

    return canvas / (canvas_weights + 1e-8)


class SAM_UNet_inference(nn.Module):
    def __init__(self, model: nn.Module):
        super(SAM_UNet_inference, self).__init__()
        model.eval()
        self.model = model

    def forward(
        self,
        input_tensor: torch.Tensor,
        window_size: Tuple[int, int] = (256, 256),
        overlap: int = 25,
        max_cell_size: int = 0,
        output_channels: int = 6,
        batch_size: int = 20
    ) -> torch.Tensor:
        """TorchScript-friendly sliding window inference."""
        h = int(input_tensor.shape[-2])
        w = int(input_tensor.shape[-1])

        # Adjust overlap if image is too small
        if 2 * (overlap + max_cell_size) >= min(h, w):
            overlap = 0
            max_cell_size = 0

        # Clamp window size to image size
        window_h = min(window_size[0], h)
        window_w = min(window_size[1], w)
        window_size = (window_h, window_w)


        # Generate tiles and indices
        tuple_index = _chops(input_tensor.shape, shape=window_size, overlap=2 * (overlap + max_cell_size))
        tile_list: List[torch.Tensor] = _tiles_from_chops(input_tensor, shape=window_size, tuple_index=tuple_index)

        print(len(tile_list), "tiles generated")
        print(tile_list[0].shape, "tile shape")

        num_tiles = len(tile_list)
        num_batches = (num_tiles + batch_size - 1) // batch_size
        label_list: List[torch.Tensor] = []

        for i in range(num_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, num_tiles)
            batch_tiles = torch.cat(tile_list[start:end])
            print(batch_tiles.shape, "batch size shape")
            with torch.no_grad():
                batch_out = self.model(batch_tiles)
            label_list.append(batch_out)

        labels = torch.cat(label_list, dim=0)

        out = _stitch_mean(
            labels,
            shape=window_size,
            chop_list=tuple_index,
            final_shape=(output_channels, h, w)
        )[None]

        print("Output shape:", out.shape)

        return out

