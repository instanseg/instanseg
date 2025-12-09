"""Tests for visualization functions in instanseg.utils.visualization"""

import numpy as np
import pytest
import torch

from instanseg.utils.visualization import (
    apply_cmap,
    label_to_color_image,
    _to_scaled_uint8,
    _to_rgb_channels_last,
    save_image_with_label_overlay,
    display_cells_and_nuclei,
)


class TestApplyCmap:
    """Tests for apply_cmap function"""

    def test_basic_colormap_application(self):
        x = np.array([[0, 1, 2], [3, 4, 5]])
        result = apply_cmap(x)
        assert result.shape == (2, 3, 3)  # RGB output
        assert result.dtype == np.uint8

    def test_with_foreground_mask(self):
        x = np.array([[0, 1, 2], [3, 4, 5]])
        fg_mask = np.array([[False, True, True], [True, True, True]])
        result = apply_cmap(x, fg_mask=fg_mask)
        # Background pixels should have bg_intensity
        assert np.all(result[0, 0] == 255)  # Default bg_intensity

    def test_custom_background_intensity(self):
        x = np.array([[0, 1], [2, 3]])
        fg_mask = np.array([[False, True], [True, True]])
        result = apply_cmap(x, fg_mask=fg_mask, bg_intensity=128)
        assert np.all(result[0, 0] == 128)


class TestLabelToColorImage:
    """Tests for label_to_color_image function"""

    def test_basic_label_conversion(self):
        labels = np.array([[0, 1, 1], [2, 2, 0]])
        result = label_to_color_image(labels)
        assert result.shape == (2, 3, 3)
        assert result.dtype == np.uint8

    def test_zero_labels_are_black(self):
        labels = np.array([[0, 1], [1, 0]])
        result = label_to_color_image(labels)
        np.testing.assert_array_equal(result[0, 0], [0, 0, 0])
        np.testing.assert_array_equal(result[1, 1], [0, 0, 0])

    def test_negative_one_labels_are_white(self):
        labels = np.array([[0, -1], [1, -1]])
        result = label_to_color_image(labels)
        np.testing.assert_array_equal(result[0, 1], [255, 255, 255])
        np.testing.assert_array_equal(result[1, 1], [255, 255, 255])

    def test_rejects_non_2d_input(self):
        labels = np.array([[[0, 1], [1, 0]]])
        with pytest.raises(ValueError):
            label_to_color_image(labels)


class TestToScaledUint8:
    """Tests for _to_scaled_uint8 function"""

    def test_basic_scaling(self):
        im = np.array([[0, 50], [100, 200]], dtype=np.float32)
        result = _to_scaled_uint8(im, clip_percentile=0)
        assert result.dtype == np.uint8
        assert result.min() >= 0
        assert result.max() <= 255

    def test_output_range(self):
        im = np.random.rand(100, 100) * 1000
        result = _to_scaled_uint8(im)
        assert result.min() >= 0
        assert result.max() <= 255


class TestToRgbChannelsLast:
    """Tests for _to_rgb_channels_last function"""

    def test_channels_first_to_last(self):
        im = np.random.rand(3, 100, 100).astype(np.float32)
        result = _to_rgb_channels_last(im, input_channels_first=True)
        assert result.shape == (100, 100, 3)
        assert result.dtype == np.uint8

    def test_2d_to_rgb(self):
        im = np.random.rand(100, 100).astype(np.float32)
        result = _to_rgb_channels_last(im)
        # 2D images get converted and repeated to make RGB
        assert result.ndim == 2 or result.shape == (100, 100, 3)

    def test_rejects_4d_input(self):
        im = np.random.rand(1, 3, 100, 100)
        with pytest.raises(ValueError):
            _to_rgb_channels_last(im)


class TestSaveImageWithLabelOverlay:
    """Tests for save_image_with_label_overlay function"""

    def test_torch_tensor_single_channel_labels(self):
        im = np.random.rand(100, 100, 3).astype(np.uint8) * 255
        lab = torch.zeros((1, 100, 100), dtype=torch.int32)
        lab[0, 20:40, 20:40] = 1
        lab[0, 60:80, 60:80] = 2
        
        result = save_image_with_label_overlay(im, lab, return_image=True)
        assert result.shape[:2] == (100, 100)
        assert result.shape[2] in [3, 4]  # RGB or RGBA

    def test_torch_tensor_two_channel_labels(self):
        im = np.random.rand(100, 100, 3).astype(np.uint8) * 255
        lab = torch.zeros((2, 100, 100), dtype=torch.int32)
        lab[0, 20:40, 20:40] = 1  # Nuclei
        lab[1, 15:45, 15:45] = 1  # Cells
        
        result = save_image_with_label_overlay(im, lab, return_image=True)
        assert result.shape[:2] == (100, 100)

    def test_numpy_labels_with_boundary_mode(self):
        im = np.random.rand(100, 100, 3).astype(np.uint8) * 255
        lab = np.zeros((100, 100), dtype=np.int32)
        lab[20:40, 20:40] = 1
        
        for mode in ['thick', 'inner', 'outer']:
            result = save_image_with_label_overlay(
                im, lab, return_image=True, label_boundary_mode=mode
            )
            assert result is not None

    def test_color_options(self):
        im = np.random.rand(100, 100, 3).astype(np.uint8) * 255
        lab = np.zeros((100, 100), dtype=np.int32)
        lab[20:40, 20:40] = 1
        
        for color in ["red", "green", "blue", "cyan", "magenta"]:
            result = save_image_with_label_overlay(
                im, lab, return_image=True, label_colors=color
            )
            assert result is not None


class TestDisplayCellsAndNuclei:
    """Tests for display_cells_and_nuclei function"""

    def test_single_channel_display(self):
        lab = torch.zeros((1, 100, 100), dtype=torch.int32)
        lab[0, 20:40, 20:40] = 1
        
        result = display_cells_and_nuclei(lab)
        assert result.shape[:2] == (100, 100)

    def test_two_channel_display(self):
        lab = torch.zeros((2, 100, 100), dtype=torch.int32)
        lab[0, 20:40, 20:40] = 1  # Nuclei
        lab[1, 15:45, 15:45] = 1  # Cells
        
        result = display_cells_and_nuclei(lab)
        assert result.shape[:2] == (100, 100)

