"""Tests for PyTorch utility functions in instanseg.utils.pytorch_utils"""

import numpy as np
import pytest
import torch

from instanseg.utils.pytorch_utils import (
    torch_fastremap,
    torch_onehot,
    fast_iou,
    remap_values,
    _to_tensor_float32,
    _to_ndim,
    connected_components,
)


class TestRemapValues:
    """Tests for remap_values function"""

    def test_basic_remapping(self):
        remapping = torch.tensor([[1, 2, 3], [10, 20, 30]])
        x = torch.tensor([[1, 2], [3, 1]])
        result = remap_values(remapping, x)
        expected = torch.tensor([[10, 20], [30, 10]])
        torch.testing.assert_close(result, expected)

    def test_preserves_shape(self):
        remapping = torch.tensor([[1, 2], [10, 20]])
        x = torch.tensor([[[1, 2, 1], [2, 1, 2]]])
        result = remap_values(remapping, x)
        assert result.shape == x.shape


class TestTorchFastremap:
    """Tests for torch_fastremap function"""

    def test_basic_remapping(self):
        x = torch.tensor([[0, 5, 5], [10, 10, 0]])
        result = torch_fastremap(x)
        # Should remap to consecutive integers starting from 0 or 1
        unique_result = torch.unique(result)
        # Check values are consecutive
        assert len(unique_result) == 3  # 0, and two non-zero labels

    def test_empty_tensor(self):
        x = torch.zeros((10, 10), dtype=torch.int64)
        result = torch_fastremap(x)
        assert result.max() == 0

    def test_single_label(self):
        x = torch.zeros((10, 10), dtype=torch.int64)
        x[2:5, 2:5] = 100
        result = torch_fastremap(x)
        assert result.max() >= 1
        assert (result > 0).sum() == (x > 0).sum()


class TestTorchOnehot:
    """Tests for torch_onehot function"""

    def test_basic_onehot(self):
        x = torch.tensor([[0, 1, 1], [2, 2, 0]])
        result = torch_onehot(x)
        # Should have 2 channels (for labels 1 and 2)
        assert result.shape[1] == 2

    def test_empty_labels(self):
        # Tests the fix for empty labels (all zeros) - should return tensor with 0 channels
        x = torch.zeros((1, 1, 10, 10), dtype=torch.int64)
        result = torch_onehot(x)
        # Empty labels should return shape (1, 0, H, W)
        assert result.shape == (1, 0, 10, 10)
        assert result.numel() == 0

    def test_single_label(self):
        x = torch.zeros((10, 10), dtype=torch.int64)
        x[2:5, 2:5] = 1
        result = torch_onehot(x)
        assert result.shape[1] == 1


class TestFastIou:
    """Tests for fast_iou function"""

    def test_identical_masks(self):
        onehot = torch.zeros((2, 10, 10))
        onehot[0, 2:5, 2:5] = 1
        onehot[1, 2:5, 2:5] = 1
        iou = fast_iou(onehot)
        # Identical masks should have IoU = 1
        assert torch.allclose(iou[0, 1], torch.tensor(1.0))

    def test_non_overlapping_masks(self):
        onehot = torch.zeros((2, 10, 10))
        onehot[0, 0:3, 0:3] = 1
        onehot[1, 7:10, 7:10] = 1
        iou = fast_iou(onehot)
        # Non-overlapping masks should have IoU = 0
        assert torch.allclose(iou[0, 1], torch.tensor(0.0))

    def test_partial_overlap(self):
        onehot = torch.zeros((2, 10, 10))
        onehot[0, 0:5, 0:5] = 1  # 25 pixels
        onehot[1, 3:8, 3:8] = 1  # 25 pixels, overlap = 4 pixels
        iou = fast_iou(onehot)
        # IoU should be between 0 and 1
        assert 0 < iou[0, 1] < 1


class TestToTensorFloat32:
    """Tests for _to_tensor_float32 function"""

    def test_numpy_to_tensor(self):
        arr = np.random.rand(3, 100, 100).astype(np.float64)
        result = _to_tensor_float32(arr)
        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.float32

    def test_tensor_passthrough(self):
        t = torch.rand(3, 100, 100)
        result = _to_tensor_float32(t)
        assert result.dtype == torch.float32

    def test_uint8_to_float32(self):
        arr = (np.random.rand(3, 100, 100) * 255).astype(np.uint8)
        result = _to_tensor_float32(arr)
        assert result.dtype == torch.float32


class TestToNdim:
    """Tests for _to_ndim function"""

    def test_expand_2d_to_3d(self):
        x = torch.rand(100, 100)
        result = _to_ndim(x, 3)
        assert result.dim() == 3

    def test_expand_3d_to_4d(self):
        x = torch.rand(3, 100, 100)
        result = _to_ndim(x, 4)
        assert result.dim() == 4

    def test_squeeze_5d_to_4d(self):
        x = torch.rand(1, 1, 3, 100, 100)
        result = _to_ndim(x, 4)
        assert result.dim() == 4


class TestConnectedComponents:
    """Tests for connected_components function"""

    def test_single_component(self):
        x = torch.zeros((1, 1, 10, 10), dtype=torch.int64)
        x[0, 0, 2:8, 2:8] = 1
        result = connected_components(x)
        # Should have exactly one unique non-zero label
        unique_nonzero = torch.unique(result[result > 0])
        assert len(unique_nonzero) == 1

    def test_multiple_components(self):
        x = torch.zeros((1, 1, 10, 10), dtype=torch.int64)
        x[0, 0, 0:3, 0:3] = 1
        x[0, 0, 7:10, 7:10] = 1
        result = connected_components(x)
        # Should have exactly two unique non-zero labels
        unique_nonzero = torch.unique(result[result > 0])
        assert len(unique_nonzero) == 2

    def test_empty_input(self):
        x = torch.zeros((1, 1, 10, 10), dtype=torch.int64)
        result = connected_components(x)
        assert result.max() == 0

    def test_connected_vs_separate(self):
        # Connected L shape
        x1 = torch.zeros((1, 1, 10, 10), dtype=torch.int64)
        x1[0, 0, 0:5, 2:4] = 1
        x1[0, 0, 4:6, 2:8] = 1
        result1 = connected_components(x1)
        unique1 = torch.unique(result1[result1 > 0])
        
        # Separate components
        x2 = torch.zeros((1, 1, 10, 10), dtype=torch.int64)
        x2[0, 0, 0:3, 0:3] = 1
        x2[0, 0, 7:10, 7:10] = 1
        result2 = connected_components(x2)
        unique2 = torch.unique(result2[result2 > 0])
        
        assert len(unique1) == 1  # One connected component
        assert len(unique2) == 2  # Two separate components

