from types import SimpleNamespace

import numpy as np
import pytest
import tifffile
import torch
import zarr

from instanseg.inference_class import (
    InstanSeg,
    _acquired_tile_grid,
    _compute_global_wsi_normalization,
    _normalise_selected_wsi_tile,
    _percentile_from_histogram,
    _uint16_histogram_for_acquired_tiles,
    _wsi_edge_metadata,
)


def test_histogram_percentiles_match_numpy_linear_percentiles():
    values = np.array([0, 0, 1, 2, 2, 2, 100, 65535], dtype=np.uint16)
    histogram = np.bincount(values, minlength=65536)

    for percentile in (0, 0.1, 25, 50, 99.9, 100):
        expected = np.percentile(values, percentile, method="linear")
        observed = _percentile_from_histogram(histogram, percentile)
        assert observed == pytest.approx(expected)


def test_acquisition_grid_excludes_only_complete_zero_tiles():
    reference = np.zeros((4, 6), dtype=np.uint16)
    reference[0, 0] = 5
    reference[2, 3] = 7
    acquired = _acquired_tile_grid(reference, (2, 3))
    assert acquired.tolist() == [[True, False], [False, True]]

    image = np.arange(24, dtype=np.uint16).reshape(4, 6)
    image[0, 1] = 0  # A real zero inside an acquired tile must remain included.
    histogram, pixel_count = _uint16_histogram_for_acquired_tiles(
        image, acquired, (2, 3)
    )
    expected = np.concatenate([image[:2, :3].ravel(), image[2:, 3:].ravel()])
    assert pixel_count == expected.size
    assert np.array_equal(histogram, np.bincount(expected, minlength=65536))
    assert histogram[0] == 2


def test_rectangular_wsi_grid_uses_row_and_column_counts_correctly():
    ignore, row_start, column_start, edge = _wsi_edge_metadata(
        i=0,
        j=2,
        n_rows=2,
        n_cols=3,
        window_i=0,
        window_j=40,
        pad=5,
    )
    assert ignore == ["top", "right"]
    assert row_start == 0
    assert column_start == 45
    assert edge

    ignore, row_start, column_start, edge = _wsi_edge_metadata(
        i=1,
        j=0,
        n_rows=2,
        n_cols=3,
        window_i=20,
        window_j=0,
        pad=5,
    )
    assert ignore == ["left", "bottom"]
    assert row_start == 25
    assert column_start == 0
    assert edge


def test_wsi_tile_channels_are_selected_before_fixed_normalization():
    tile = np.zeros((5, 6, 4), dtype=np.uint16)
    tile[..., 1] = 15
    tile[..., 3] = 35
    normalized = _normalise_selected_wsi_tile(
        tile,
        channel_ids=[3, 1],
        bounds=[(30, 40), (10, 20)],
    )
    assert normalized.dtype == torch.float32
    assert normalized.shape == (2, 5, 6)
    assert torch.all(normalized == 0.5)


def _write_test_ome(path, image):
    tifffile.imwrite(
        path,
        image,
        metadata={"axes": "CYX", "PhysicalSizeX": 0.5, "PhysicalSizeXUnit": "µm"},
        photometric="minisblack",
        tile=(16, 16),
    )


def test_global_normalization_reads_one_uint16_channel_and_ignores_empty_tiles(tmp_path):
    image = np.zeros((2, 32, 32), dtype=np.uint16)
    image[0, :16, :16] = 10
    image[0, :16, 16:] = 20
    image[0, 16:, :16] = 30
    # The lower-right reference tile remains zero and is unacquired.
    image[1, :16, :16] = 100
    image[1, :16, 16:] = 200
    image[1, 16:, :16] = 300
    image[1, 16:, 16:] = 65535  # Must be excluded with the unacquired tile.
    path = tmp_path / "test.ome.tif"
    _write_test_ome(path, image)

    result = _compute_global_wsi_normalization(
        path,
        channel_ids=[1, 0],
        reference_channel_id=0,
        percentiles=(0.1, 99.9),
    )

    assert result["channel_ids"] == [1, 0]
    assert result["acquisition_tiles"]["included"] == 3
    assert result["acquisition_tiles"]["excluded"] == 1
    assert result["channel_pixel_counts"] == [3 * 16 * 16, 3 * 16 * 16]
    expected_channel_1 = np.concatenate(
        [
            image[1, :16, :16].ravel(),
            image[1, :16, 16:].ravel(),
            image[1, 16:, :16].ravel(),
        ]
    )
    assert result["bounds"][0] == pytest.approx(
        np.percentile(expected_channel_1, [0.1, 99.9], method="linear")
    )


def test_read_slide_imports_tiffslide_without_external_module_patch(tmp_path):
    image = np.ones((2, 32, 32), dtype=np.uint16)
    path = tmp_path / "slide.ome.tif"
    _write_test_ome(path, image)
    inst = object.__new__(InstanSeg)
    inst.prefered_image_reader = "tiffslide"

    slide = inst.read_slide(str(path))

    assert slide.dimensions == (32, 32)


class _FakeSlide:
    def __init__(self, image):
        self.image = np.moveaxis(image, 0, -1)
        self.dimensions = (image.shape[2], image.shape[1])
        self.level_downsamples = [1.0]

    def get_best_level_for_downsample(self, _scale):
        return 0

    def read_region(self, location, level, size, as_array=True):
        assert level == 0 and as_array
        x0, y0 = location
        width, height = size
        output = np.zeros((height, width, self.image.shape[-1]), dtype=self.image.dtype)
        x1 = min(self.image.shape[1], x0 + width)
        y1 = min(self.image.shape[0], y0 + height)
        output[: y1 - y0, : x1 - x0] = self.image[y0:y1, x0:x1]
        return output


def test_global_normalized_wsi_writes_completed_two_plane_zarr(tmp_path):
    image = np.ones((3, 48, 80), dtype=np.uint16)
    image[0] *= 10
    image[1] *= 20
    image[2] *= 30
    image_path = tmp_path / "rectangular.ome.tif"
    _write_test_ome(image_path, image)

    inst = object.__new__(InstanSeg)
    inst.instanseg = SimpleNamespace(cells_and_nuclei=True, pixel_size=0.5)
    inst.prediction_tag = "_prediction"
    inst.verbose = False
    inst.read_image = lambda path, processing_method: (str(path), 0.5)
    inst.read_slide = lambda path: _FakeSlide(image)

    def fake_eval_small(input_tensor, **kwargs):
        assert input_tensor.shape[0] == 2
        assert kwargs["normalise"] is False
        height, width = input_tensor.shape[-2:]
        labels = torch.zeros((1, 2, height, width), dtype=torch.int32)
        labels[0, 0, 10:14, 10:14] = 1
        labels[0, 1, 8:16, 8:16] = 1
        return labels

    inst.eval_small_image = fake_eval_small
    output_path = tmp_path / "prediction.zarr"
    observed = inst.eval_whole_slide_image_global_normalization(
        image_path,
        channel_ids=[2, 0],
        reference_channel_id=0,
        tile_size=32,
        overlap=2,
        detection_size=2,
        output_path=output_path,
    )

    assert observed == output_path
    output = zarr.open(str(output_path), mode="r")
    assert output.shape == (2, 48, 80)
    assert output.chunks == (1, 32, 32)
    assert output.dtype == np.int32
    assert output.attrs["status"] == "complete"
    assert output.attrs["channel_ids"] == [2, 0]
    assert output.attrs["planes"] == ["nuclei", "cells"]
    assert output.attrs["wsi_settings"] == {
        "tile_size": 32,
        "overlap": 2,
        "detection_size": 2,
        "resolve_cell_and_nucleus": None,
    }
    assert np.count_nonzero(output[0]) > 0
    assert np.count_nonzero(output[1]) > 0
    assert (tmp_path / "prediction.zarr.normalization.json").is_file()
