import torch
from instanseg.utils.pytorch_utils import torch_onehot, fast_dual_iou, torch_sparse_onehot, fast_sparse_dual_iou

def test_iou():
    out = torch.randint(0, 50, (1, 2, 124, 256), dtype=torch.float32)
    onehot1 = torch_onehot(out[0, 0])[0]
    onehot2 = torch_onehot(out[0, 1])[0]
    iou_dense = fast_dual_iou(onehot1, onehot2)

    onehot1 = torch_sparse_onehot(out[0, 0], flatten=True)[0]
    onehot2 = torch_sparse_onehot(out[0, 1], flatten=True)[0]
    iou_sparse = fast_sparse_dual_iou(onehot1, onehot2)

    assert torch.allclose(iou_dense, iou_sparse)
