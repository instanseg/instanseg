
def test_tiling():

    import torch
    from instanseg.utils.tiling import _chops, _tiles_from_chops, _stitch
    from instanseg.utils.pytorch_utils import torch_sparse_onehot, fast_sparse_dual_iou, connected_components

    torch.random.manual_seed(0)
    input_tensor = torch.randint(0,2,(1,1, 512, 256))
    input_tensor = connected_components(input_tensor)

    overlap = 10
    max_cell_size = 20
    window_size = (70,70)

    tuple_index = _chops(input_tensor.shape, shape=window_size, overlap=2 * (overlap + max_cell_size))
    tile_list = _tiles_from_chops(input_tensor, shape=window_size, tuple_index=tuple_index)
    assert len(tile_list) == len(tuple_index[0]) * len(tuple_index[1])
    labels_list = [lab for lab in tile_list]
    output, _ = _stitch([lab[0,0] for lab in labels_list],
                                shape=window_size,
                                chop_list=tuple_index,
                                offset = overlap,
                                final_shape=(1, input_tensor.shape[-2], input_tensor.shape[-1]))

    assert output.shape[-2:] == input_tensor.shape[-2:]

    out = torch.stack([input_tensor[0], output], dim=1)
    onehot1 = torch_sparse_onehot(out[0, 0], flatten=True)[0]
    onehot2 = torch_sparse_onehot(out[0, 1], flatten=True)[0]
    iou_sparse = fast_sparse_dual_iou(onehot1, onehot2)

    assert iou_sparse.shape[-1] == iou_sparse.shape[-2] and iou_sparse.sum() == iou_sparse.shape[-1]
    assert iou_sparse.shape[-1] == len(torch.unique(input_tensor[input_tensor > 0]))
    assert (iou_sparse.sum(0) == iou_sparse.sum(1)).all()

def test_tiling_cross_channel_mapping():
    """
    Ensure stitching with map_list aligns the existing nucleus to its cell
    and does not create mismatches.
    """

    import torch
    import torch.nn.functional as F
    from instanseg.utils.tiling import _chops, _tiles_from_chops, _stitch

    # ---- tiling config ----
    H, W = 100, 100
    window = (40, 40)      # vertical boundary at x = 40
    offset = 6
    max_cell_size = 12
    overlap = 2 * (offset + max_cell_size)  # 36 < 40

    cell_size = max_cell_size - 3
    nucleus_size = max_cell_size - 5

    n = 10

    cells  = torch.zeros((1,1,H,W), dtype=torch.int32)
    nuclei = torch.zeros_like(cells)

    # random seed locations
    rng = torch.Generator(device=cells.device).manual_seed(42)

    ys = torch.randint(0, H, (n,), generator=rng, device=cells.device)
    xs = torch.randint(0, W, (n,), generator=rng, device=cells.device)

    rc = cell_size   // 2
    rn = nucleus_size// 2

    for lab, (y, x) in enumerate(zip(ys.tolist(), xs.tolist()), start=1):
        # paint cell square
        y0, y1 = max(0, y - rc), min(H, y + rc + 1)
        x0, x1 = max(0, x - rc), min(W, x + rc + 1)
        cells[0,0, y0:y1, x0:x1] = lab

        # optionally paint nucleus square with SAME label
        if torch.rand(()) < 0.5:
            y0n, y1n = max(0, y - rn), min(H, y + rn + 1)
            x0n, x1n = max(0, x - rn), min(W, x + rn + 1)
            nuclei[0,0, y0n:y1n, x0n:x1n] = lab


    # tiles
    chops = _chops(cells.shape, shape=window, overlap=overlap)
    tiles_cells = _tiles_from_chops(cells, shape=window, tuple_index=chops)
    tiles_nuclei = _tiles_from_chops(nuclei, shape=window, tuple_index=chops)

    # stitch cells
    stitched_cells, map_list = _stitch(
        [t[0, 0] for t in tiles_cells],
        shape=window,
        chop_list=chops,
        offset=offset,
        final_shape=(1, H, W),
    )

    # stitch nuclei
    stitched_nuclei, _ = _stitch(
        [t[0, 0] for t in tiles_nuclei],
        shape=window,
        chop_list=chops,
        offset=offset,
        final_shape=(1, H, W),
        map_list=map_list,
    )

    # assertions
    for y, x in zip(ys.tolist(), xs.tolist()):
        
        cell = stitched_cells[0, y, x]
        nucleus = stitched_nuclei[0, y, x] 
        if nucleus:
            assert nucleus == cell, f"Nucleus {nucleus} inside cell {cell}."
