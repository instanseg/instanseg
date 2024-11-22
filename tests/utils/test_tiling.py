def test_tiling():

    import torch
    from instanseg.utils.tiling import _chops, _tiles_from_chops, _stitch
    from instanseg.utils.pytorch_utils import (
        torch_sparse_onehot,
        fast_sparse_dual_iou,
        connected_components,
    )

    torch.random.manual_seed(0)
    input_tensor = torch.randint(0, 2, (1, 1, 512, 256))
    input_tensor = connected_components(input_tensor)

    overlap = 10
    max_cell_size = 20
    window_size = (70, 70)

    tuple_index = _chops(
        input_tensor.shape, shape=window_size, overlap=2 * (overlap + max_cell_size)
    )
    tile_list = _tiles_from_chops(
        input_tensor, shape=window_size, tuple_index=tuple_index
    )
    assert len(tile_list) == len(tuple_index[0]) * len(tuple_index[1])
    labels_list = [lab for lab in tile_list]
    output = _stitch(
        [lab[0, 0] for lab in labels_list],
        shape=window_size,
        chop_list=tuple_index,
        offset=overlap,
        final_shape=(1, input_tensor.shape[-2], input_tensor.shape[-1]),
    )

    assert output.shape[-2:] == input_tensor.shape[-2:]

    out = torch.stack([input_tensor[0], output], dim=1)
    onehot1 = torch_sparse_onehot(out[0, 0], flatten=True)[0]
    onehot2 = torch_sparse_onehot(out[0, 1], flatten=True)[0]
    iou_sparse = fast_sparse_dual_iou(onehot1, onehot2)

    assert (
        iou_sparse.shape[-1] == iou_sparse.shape[-2]
        and iou_sparse.sum() == iou_sparse.shape[-1]
    )
    assert iou_sparse.shape[-1] == len(torch.unique(input_tensor[input_tensor > 0]))
    assert (iou_sparse.sum(0) == iou_sparse.sum(1)).all()
