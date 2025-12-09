"""Edge case and integration tests for InstanSeg"""

import numpy as np
import pytest
import sys
import torch
from pathlib import Path

# Determine device
device = "cuda" if torch.cuda.is_available() else "cpu"


class TestInputShapes:
    """Tests for various input shapes and edge cases"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup InstanSeg model for tests"""
        from instanseg import InstanSeg
        self.instanseg = InstanSeg("fluorescence_nuclei_and_cells", verbosity=0, device=device)

    def test_single_channel_input(self):
        """Test single channel input"""
        input_tensor = torch.rand(1, 1, 256, 256)
        result = self.instanseg.eval_small_image(input_tensor, pixel_size=0.5, return_image_tensor=False)
        assert result is not None
        assert result.shape[-2:] == (256, 256)

    def test_multichannel_input(self):
        """Test multi-channel input (more than 3 channels)"""
        input_tensor = torch.rand(1, 5, 256, 256)
        result = self.instanseg.eval_small_image(input_tensor, pixel_size=0.5, return_image_tensor=False)
        assert result is not None

    def test_non_square_input(self):
        """Test non-square input"""
        input_tensor = torch.rand(1, 3, 128, 512)
        result = self.instanseg.eval_small_image(input_tensor, pixel_size=0.5, return_image_tensor=False)
        assert result.shape[-2:] == (128, 512)

    def test_small_input(self):
        """Test very small input"""
        input_tensor = torch.rand(1, 3, 64, 64)
        result = self.instanseg.eval_small_image(input_tensor, pixel_size=0.5, return_image_tensor=False)
        assert result is not None

    def test_numpy_input(self):
        """Test numpy array input"""
        input_array = np.random.rand(3, 256, 256).astype(np.float32)
        result = self.instanseg.eval_small_image(input_array, pixel_size=0.5, return_image_tensor=False)
        assert result is not None

    def test_different_pixel_sizes(self):
        """Test different pixel sizes"""
        input_tensor = torch.rand(1, 3, 256, 256)
        
        for pixel_size in [0.25, 0.5, 1.0]:
            result = self.instanseg.eval_small_image(
                input_tensor, pixel_size=pixel_size, 
                return_image_tensor=False, rescale_output=True
            )
            # Output should match input size when rescale_output=True
            assert result.shape[-2:] == (256, 256)

    def test_no_pixel_size(self):
        """Test without pixel size (should still work with warning)"""
        input_tensor = torch.rand(1, 3, 256, 256)
        result = self.instanseg.eval_small_image(
            input_tensor, pixel_size=None, return_image_tensor=False
        )
        assert result is not None


class TestNormalization:
    """Tests for normalization options"""

    @pytest.fixture(autouse=True)
    def setup(self):
        from instanseg import InstanSeg
        self.instanseg = InstanSeg("fluorescence_nuclei_and_cells", verbosity=0, device=device)

    def test_with_normalization(self):
        input_tensor = torch.rand(1, 3, 256, 256) * 255
        result = self.instanseg.eval_small_image(
            input_tensor, pixel_size=0.5, normalise=True, return_image_tensor=False
        )
        assert result is not None

    def test_without_normalization(self):
        # Pre-normalized input (0-1 range)
        input_tensor = torch.rand(1, 3, 256, 256)
        result = self.instanseg.eval_small_image(
            input_tensor, pixel_size=0.5, normalise=False, return_image_tensor=False
        )
        assert result is not None


class TestTargetSegmentation:
    """Tests for different target segmentation modes"""

    @pytest.fixture(autouse=True)
    def setup(self):
        from instanseg import InstanSeg
        self.instanseg = InstanSeg("fluorescence_nuclei_and_cells", verbosity=0, device=device)

    def test_nuclei_only(self):
        input_tensor = torch.rand(1, 3, 256, 256)
        result = self.instanseg.eval_small_image(
            input_tensor, pixel_size=0.5, target="nuclei", return_image_tensor=False
        )
        assert result.shape[1] == 1  # Only nuclei channel

    def test_cells_only(self):
        input_tensor = torch.rand(1, 3, 256, 256)
        result = self.instanseg.eval_small_image(
            input_tensor, pixel_size=0.5, target="cells", return_image_tensor=False
        )
        assert result.shape[1] == 1  # Only cells channel

    def test_all_outputs(self):
        input_tensor = torch.rand(1, 3, 256, 256)
        result = self.instanseg.eval_small_image(
            input_tensor, pixel_size=0.5, target="all_outputs", return_image_tensor=False
        )
        assert result.shape[1] == 2  # Both nuclei and cells


class TestMediumImageTiling:
    """Tests for medium image tiling functionality"""

    @pytest.fixture(autouse=True)
    def setup(self):
        from instanseg import InstanSeg
        self.instanseg = InstanSeg("fluorescence_nuclei_and_cells", verbosity=0, device=device)

    def test_tiled_processing(self):
        """Test that tiled processing produces consistent output shape"""
        input_tensor = torch.rand(1, 3, 1024, 1024)
        result = self.instanseg.eval_medium_image(
            input_tensor, pixel_size=0.5, tile_size=256, 
            return_image_tensor=False, rescale_output=True
        )
        assert result.shape[-2:] == (1024, 1024)

    def test_different_tile_sizes(self):
        """Test different tile sizes"""
        input_tensor = torch.rand(1, 3, 512, 512)
        
        # Test with medium and large tile sizes (small tiles can be slow)
        for tile_size in [256, 512]:
            result = self.instanseg.eval_medium_image(
                input_tensor, pixel_size=0.5, tile_size=tile_size,
                return_image_tensor=False, rescale_output=True
            )
            assert result.shape[-2:] == (512, 512)


class TestDisplayFunction:
    """Tests for display functionality"""

    @pytest.fixture(autouse=True)
    def setup(self):
        from instanseg import InstanSeg
        self.instanseg = InstanSeg("fluorescence_nuclei_and_cells", verbosity=0, device=device)

    def test_display_with_numpy_input(self):
        image = np.random.rand(3, 256, 256).astype(np.float32)
        labels = torch.zeros((1, 2, 256, 256), dtype=torch.int32)
        labels[0, 0, 50:100, 50:100] = 1
        labels[0, 1, 45:105, 45:105] = 1
        
        result = self.instanseg.display(image, labels)
        assert result is not None
        assert result.shape[:2] == (256, 256)

    def test_display_with_tensor_input(self):
        image = torch.rand(3, 256, 256)
        labels = torch.zeros((1, 2, 256, 256), dtype=torch.int32)
        labels[0, 0, 50:100, 50:100] = 1
        labels[0, 1, 45:105, 45:105] = 1
        
        result = self.instanseg.display(image, labels)
        assert result is not None

    def test_display_single_channel_labels(self):
        image = torch.rand(3, 256, 256)
        labels = torch.zeros((1, 1, 256, 256), dtype=torch.int32)
        labels[0, 0, 50:100, 50:100] = 1
        labels[0, 0, 150:200, 150:200] = 2
        
        result = self.instanseg.display(image, labels)
        assert result is not None


class TestChannelSelection:
    """Tests for channel selection functionality"""

    @pytest.fixture(autouse=True)
    def setup(self):
        from instanseg import InstanSeg
        self.instanseg = InstanSeg("fluorescence_nuclei_and_cells", verbosity=0, device=device)

    def test_channel_ids_selection(self):
        """Test selecting specific channels from multi-channel input"""
        input_tensor = torch.rand(1, 10, 256, 256)
        result = self.instanseg.eval_small_image(
            input_tensor, pixel_size=0.5, 
            channel_ids=[0, 1, 2],
            return_image_tensor=False
        )
        assert result is not None

