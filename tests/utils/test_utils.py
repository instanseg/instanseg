"""Tests for core utility functions in instanseg.utils.utils"""

import numpy as np
import pytest
import torch

from instanseg.utils.utils import (
    _choose_device,
    _move_channel_axis,
    count_instances,
    generate_colors,
    percentile_normalize,
    _filter_kwargs,
    _estimate_image_modality,
)


class TestMoveChannelAxis:
    """Tests for _move_channel_axis function"""

    def test_numpy_channels_first_to_last(self):
        img = np.random.rand(3, 100, 100)
        result = _move_channel_axis(img, to_back=True)
        assert result.shape == (100, 100, 3)

    def test_numpy_channels_last_to_first(self):
        img = np.random.rand(100, 100, 3)
        result = _move_channel_axis(img, to_back=False)
        assert result.shape == (3, 100, 100)

    def test_torch_channels_first_to_last(self):
        img = torch.rand(3, 100, 100)
        result = _move_channel_axis(img, to_back=True)
        assert result.shape == (100, 100, 3)

    def test_torch_channels_last_to_first(self):
        img = torch.rand(100, 100, 3)
        result = _move_channel_axis(img, to_back=False)
        assert result.shape == (3, 100, 100)

    def test_2d_numpy_input(self):
        img = np.random.rand(100, 100)
        result = _move_channel_axis(img)
        assert result.shape == (1, 100, 100)

    def test_2d_torch_input(self):
        img = torch.rand(100, 100)
        result = _move_channel_axis(img)
        assert result.shape == (1, 100, 100)

    def test_4d_numpy_gets_squeezed(self):
        # 4D input with size 1 dimension gets squeezed
        img = np.random.rand(1, 3, 100, 100)
        result = _move_channel_axis(img)
        assert result.ndim == 3


class TestPercentileNormalize:
    """Tests for percentile_normalize function"""

    def test_numpy_2d_normalization(self):
        img = np.random.rand(100, 100) * 255
        result = percentile_normalize(img)
        # After normalization, most values should be between 0 and 1
        assert result.min() >= -0.1
        assert result.max() <= 1.1

    def test_numpy_3d_normalization(self):
        img = np.random.rand(3, 100, 100) * 255
        result = percentile_normalize(img)
        assert result.shape == img.shape

    def test_torch_2d_normalization(self):
        img = torch.rand(100, 100) * 255
        result = percentile_normalize(img)
        assert isinstance(result, torch.Tensor)

    def test_torch_3d_normalization(self):
        img = torch.rand(3, 100, 100) * 255
        result = percentile_normalize(img)
        assert result.shape == img.shape

    def test_subsampling_factor(self):
        img = np.random.rand(100, 100) * 255
        result = percentile_normalize(img, subsampling_factor=2)
        # Result may have extra dimension from atleast_3d
        assert result.shape[:2] == img.shape[:2]

    def test_custom_percentile(self):
        img = np.random.rand(100, 100) * 255
        result = percentile_normalize(img, percentile=1.0)
        assert result is not None


class TestGenerateColors:
    """Tests for generate_colors function"""

    def test_generates_correct_number(self):
        colors = generate_colors(5)
        assert len(colors) == 5

    def test_colors_are_rgb(self):
        colors = generate_colors(3)
        for color in colors:
            assert len(color) == 3
            assert all(0 <= c <= 1 for c in color)

    def test_single_color(self):
        colors = generate_colors(1)
        assert len(colors) == 1

    def test_many_colors(self):
        colors = generate_colors(100)
        assert len(colors) == 100


class TestCountInstances:
    """Tests for count_instances function"""

    def test_numpy_labeled_image(self):
        labels = np.array([[0, 0, 1, 1],
                          [0, 2, 2, 1],
                          [3, 2, 0, 0]])
        count = count_instances(labels)
        assert count == 3  # labels 1, 2, 3

    def test_torch_labeled_image(self):
        labels = torch.tensor([[0, 0, 1, 1],
                              [0, 2, 2, 1],
                              [3, 2, 0, 0]])
        count = count_instances(labels)
        assert count == 3

    def test_empty_labels(self):
        labels = np.zeros((10, 10), dtype=np.int32)
        count = count_instances(labels)
        assert count == 0

    def test_single_instance(self):
        labels = np.ones((10, 10), dtype=np.int32)
        count = count_instances(labels)
        assert count == 1


class TestChooseDevice:
    """Tests for _choose_device function"""

    def test_cpu_always_available(self):
        device = _choose_device("cpu", verbose=False)
        assert device == "cpu"

    def test_default_device_selection(self):
        device = _choose_device(None, verbose=False)
        assert device in ["cpu", "cuda", "mps"]

    def test_unavailable_cuda_fallback(self):
        if not torch.cuda.is_available():
            device = _choose_device("cuda", verbose=False)
            # Should fall back to another device
            assert device in ["cpu", "mps", None]


class TestEstimateImageModality:
    """Tests for _estimate_image_modality function"""

    def test_bright_objects_fluorescence(self):
        # Fluorescence: bright objects on dark background
        img = np.zeros((3, 100, 100))
        mask = np.zeros((100, 100))
        mask[40:60, 40:60] = 1
        img[:, 40:60, 40:60] = 1.0  # Bright objects
        
        modality = _estimate_image_modality(img, mask)
        assert modality == "Fluorescence"

    def test_dark_objects_brightfield(self):
        # Brightfield: dark objects on bright background
        img = np.ones((3, 100, 100))
        mask = np.zeros((100, 100))
        mask[40:60, 40:60] = 1
        img[:, 40:60, 40:60] = 0.2  # Dark objects
        
        modality = _estimate_image_modality(img, mask)
        assert modality == "Brightfield"

    def test_empty_mask_returns_fluorescence(self):
        img = np.random.rand(3, 100, 100)
        mask = np.zeros((100, 100))
        
        modality = _estimate_image_modality(img, mask)
        assert modality == "Fluorescence"

