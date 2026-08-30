from types import SimpleNamespace

import numpy as np
import pytest
import tifffile
import torch
import zarr

import instanseg.inference_class as inference_class
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


class _RecordingSlide:
    def __init__(self, path):
        from tiffslide import TiffSlide

        self._slide = TiffSlide(path)
        self.dimensions = self._slide.dimensions
        self.level_dimensions = self._slide.level_dimensions
        self.level_downsamples = self._slide.level_downsamples
        self.reads = []

    def get_best_level_for_downsample(self, scale):
        return self._slide.get_best_level_for_downsample(scale)

    def _read_region_loc_transform(self, location, level):
        return self._slide._read_region_loc_transform(location, level)

    def read_region(self, location, level, size, as_array=True):
        self.reads.append((location, level, size, as_array))
        return self._slide.read_region(location, level, size, as_array=as_array)

    def close(self):
        self._slide.close()


def _configure_capture_instance(image_path, slide, model_pixel_size, captured_inputs):
    inst = object.__new__(InstanSeg)
    inst.instanseg = SimpleNamespace(
        cells_and_nuclei=True, pixel_size=model_pixel_size
    )
    inst.prediction_tag = "_prediction"
    inst.verbose = False
    inst.read_image = lambda path, processing_method: (str(path), 0.5)
    inst.read_slide = lambda path: slide

    def fake_eval_small(input_tensor, **kwargs):
        captured_inputs.append(np.asarray(input_tensor))
        height, width = input_tensor.shape[-2:]
        return torch.zeros((1, 2, height, width), dtype=torch.float32)

    inst.eval_small_image = fake_eval_small
    return inst


def _configure_batch_capture_instance(image_path, calls):
    from tiffslide import TiffSlide

    class FakeModel:
        pixel_size = 0.5
        cells_and_nuclei = True
        graph = "graph"

        def __call__(self, input_tensor, **kwargs):
            return make_predictions(input_tensor, True)

    def make_predictions(input_tensor, input_was_batched):
        batch = input_tensor if input_was_batched else input_tensor.unsqueeze(0)
        calls.append((input_was_batched, batch.detach().cpu().clone()))
        height, width = batch.shape[-2:]
        labels = torch.zeros(
            (batch.shape[0], 2, height, width), dtype=torch.int32
        )
        for batch_index in range(batch.shape[0]):
            label = int(round(float(batch[batch_index, 0].mean()) * 1000)) + 1
            labels[batch_index, 0, 4:8, 4:8] = label
            labels[batch_index, 1, 6:10, 6:10] = label
        return labels

    inst = object.__new__(InstanSeg)
    inst.instanseg = FakeModel()
    inst.prediction_tag = "_prediction"
    inst.verbose = False
    inst.prefered_image_reader = "tiffslide"
    inst.inference_device = torch.device("cpu")
    inst.read_image = lambda path, processing_method: (str(path), 0.5)
    inst.read_slide = lambda path: TiffSlide(path)

    def fake_eval_small(input_tensor, **kwargs):
        return make_predictions(input_tensor, False)

    inst.eval_small_image = fake_eval_small
    return inst


def _expected_selected_regions(
    path, channel_ids, source_pixel_size, model_pixel_size, tile_size
):
    from itertools import product

    from tiffslide import TiffSlide
    from instanseg.utils.tiling import _chops

    slide = TiffSlide(path)
    scale_factor = model_pixel_size / source_pixel_size
    best_level = slide.get_best_level_for_downsample(scale_factor)
    downsample_factor = slide.level_downsamples[best_level]
    dims = (
        round(slide.dimensions[1] / scale_factor),
        round(slide.dimensions[0] / scale_factor),
    )
    shape = (tile_size, tile_size)
    chops = _chops(dims, shape, overlap=2)
    intermediate_to_final = source_pixel_size * downsample_factor / model_pixel_size
    intermediate_shape = (
        round(tile_size / intermediate_to_final),
        round(tile_size / intermediate_to_final),
    )
    regions = []
    for window_i, window_j in product(chops[0], chops[1]):
        location = (
            round(window_j * scale_factor),
            round(window_i * scale_factor),
        )
        region = slide.read_region(
            location,
            best_level,
            (intermediate_shape[1], intermediate_shape[0]),
            as_array=True,
        )
        regions.append(region[..., channel_ids])
    slide.close()
    return best_level, regions


@pytest.mark.parametrize("model_pixel_size", [0.5, 0.65])
def test_global_wsi_selected_regions_match_tiffslide_interior_and_padding(
    tmp_path, monkeypatch, model_pixel_size
):
    image = np.zeros((3, 40, 50), dtype=np.uint16)
    image[0] = np.arange(image[0].size, dtype=np.uint16).reshape(image[0].shape) + 1
    image[1] = image[0] + 1000
    image[2] = image[0] + 2000
    image_path = tmp_path / f"selected_{model_pixel_size}.ome.tif"
    _write_test_ome(image_path, image)

    slide = _RecordingSlide(image_path)
    captured_inputs = []
    inst = _configure_capture_instance(
        image_path, slide, model_pixel_size, captured_inputs
    )
    observed_normalizer_inputs = []
    original_normalizer = inference_class._normalise_selected_wsi_tile

    def capture_normalizer(input_data, selected_ids, bounds):
        observed_normalizer_inputs.append(
            (np.array(input_data), list(selected_ids))
        )
        return original_normalizer(input_data, selected_ids, bounds)

    monkeypatch.setattr(
        inference_class,
        "_normalise_selected_wsi_tile",
        capture_normalizer,
    )
    output_path = tmp_path / f"prediction_{model_pixel_size}.zarr"
    inst.eval_whole_slide_image_global_normalization(
        image_path,
        channel_ids=[2, 0],
        reference_channel_id=0,
        pixel_size=0.5,
        tile_size=16,
        overlap=0,
        detection_size=1,
        output_path=output_path,
    )

    best_level, expected_regions = _expected_selected_regions(
        image_path, [2, 0], 0.5, model_pixel_size, 16
    )
    assert slide.reads == []
    assert [ids for _, ids in observed_normalizer_inputs] == [
        [0, 1]
    ] * len(expected_regions)
    for (observed, _), expected in zip(observed_normalizer_inputs, expected_regions):
        assert np.array_equal(observed, expected)
    assert best_level == 0
    if model_pixel_size == 0.65:
        assert any(np.any(region[-1] == 0) for region in expected_regions)
    slide.close()


def _write_test_pyramid(path, image):
    with tifffile.TiffWriter(path) as writer:
        writer.write(
            image,
            subifds=1,
            metadata={
                "axes": "CYX",
                "PhysicalSizeX": 0.25,
                "PhysicalSizeXUnit": "µm",
            },
            photometric="minisblack",
            tile=(16, 16),
        )
        writer.write(image[:, ::2, ::2], subfiletype=1, tile=(16, 16))


def test_global_wsi_pyramid_reads_use_tiffslide_coordinate_conversion(
    tmp_path, monkeypatch
):
    image = np.zeros((3, 64, 80), dtype=np.uint16)
    image[0] = np.arange(image[0].size, dtype=np.uint16).reshape(image[0].shape) + 1
    image[1] = image[0] + 1000
    image[2] = image[0] + 2000
    image_path = tmp_path / "pyramid.ome.tif"
    _write_test_pyramid(image_path, image)

    slide = _RecordingSlide(image_path)
    captured_inputs = []
    inst = _configure_capture_instance(image_path, slide, 0.5, captured_inputs)
    observed_normalizer_inputs = []
    original_normalizer = inference_class._normalise_selected_wsi_tile

    def capture_normalizer(input_data, selected_ids, bounds):
        observed_normalizer_inputs.append(np.array(input_data))
        return original_normalizer(input_data, selected_ids, bounds)

    monkeypatch.setattr(
        inference_class,
        "_normalise_selected_wsi_tile",
        capture_normalizer,
    )
    inst.eval_whole_slide_image_global_normalization(
        image_path,
        channel_ids=[2, 0],
        reference_channel_id=0,
        pixel_size=0.25,
        tile_size=16,
        overlap=0,
        detection_size=1,
        output_path=tmp_path / "pyramid_prediction.zarr",
    )

    best_level, expected_regions = _expected_selected_regions(
        image_path, [2, 0], 0.25, 0.5, 16
    )
    assert best_level == 1
    assert slide.reads == []
    assert all(
        np.array_equal(observed, expected)
        for observed, expected in zip(observed_normalizer_inputs, expected_regions)
    )
    slide.close()


def test_global_wsi_unsupported_layout_falls_back_to_tiffslide(
    tmp_path, monkeypatch
):
    image = np.zeros((3, 20, 24), dtype=np.uint16)
    image[0] = 10
    image[1] = 20
    image[2] = 30
    image_path = tmp_path / "unsupported.ome.tif"
    image_path.touch()

    class UnsupportedTiffFile:
        opened = []

        def __init__(self, path):
            self.series = [
                SimpleNamespace(
                    axes="YXC",
                    levels=[SimpleNamespace(axes="YXC")],
                )
            ]
            self.closed = False
            self.opened.append(self)

        def close(self):
            self.closed = True

    monkeypatch.setattr("tifffile.TiffFile", UnsupportedTiffFile)
    monkeypatch.setattr(
        inference_class,
        "_compute_global_wsi_normalization",
        lambda *args, **kwargs: {
            "channel_ids": [2, 0],
            "reference_channel_id": 0,
            "percentiles": [0.1, 99.9],
            "bounds": [(0.0, 1.0), (0.0, 1.0)],
            "channel_pixel_counts": [1, 1],
            "source_shape": [20, 24],
            "source_dtype": "uint16",
            "source_tile_shape": [16, 16],
            "acquisition_tiles": {
                "rows": 2,
                "columns": 2,
                "included": 4,
                "excluded": 0,
            },
        },
    )

    slide = _FakeSlide(image)
    slide.level_dimensions = [slide.dimensions]
    slide.reads = []
    original_read_region = slide.read_region

    def recording_read_region(*args, **kwargs):
        slide.reads.append((args, kwargs))
        return original_read_region(*args, **kwargs)

    slide.read_region = recording_read_region
    captured_inputs = []
    inst = _configure_capture_instance(image_path, slide, 0.5, captured_inputs)
    inst.eval_whole_slide_image_global_normalization(
        image_path,
        channel_ids=[2, 0],
        reference_channel_id=0,
        tile_size=16,
        overlap=0,
        detection_size=1,
        output_path=tmp_path / "unsupported_prediction.zarr",
    )

    assert len(slide.reads) == 4
    assert captured_inputs[0].shape == (2, 16, 16)
    assert np.all(captured_inputs[0][0, :20, :24] == 30)
    assert UnsupportedTiffFile.opened[0].closed


def test_global_wsi_batching_preserves_order_and_stitched_output(tmp_path):
    image = np.zeros((3, 40, 50), dtype=np.uint16)
    image[0] = np.arange(image[0].size, dtype=np.uint16).reshape(image[0].shape) + 1
    image[1] = image[0] + 1000
    image[2] = image[0] + 2000
    image_path = tmp_path / "batched.ome.tif"
    _write_test_ome(image_path, image)

    def run(batch_size):
        calls = []
        inst = _configure_batch_capture_instance(image_path, calls)
        output_path = tmp_path / f"prediction_batch_{batch_size}.zarr"
        inst.eval_whole_slide_image_global_normalization(
            image_path,
            channel_ids=[2, 0],
            reference_channel_id=0,
            tile_size=16,
            overlap=0,
            detection_size=1,
            batch_size=batch_size,
            output_path=output_path,
        )
        output = np.asarray(zarr.open(str(output_path), mode="r"))
        return output, calls

    unbatched_output, unbatched_calls = run(1)
    batch_size = len(unbatched_calls) - 1
    batched_output, batched_calls = run(batch_size)

    assert all(not was_batched for was_batched, _ in unbatched_calls)
    assert all(was_batched for was_batched, _ in batched_calls)
    assert all(
        inputs.ndim == 4 and inputs.shape[1:] == (2, 16, 16)
        for _, inputs in batched_calls
    )
    assert batched_calls[0][1].shape[0] == batch_size
    assert len(batched_calls) < len(unbatched_calls)
    assert batched_calls[-1][1].shape[0] == 1  # non-divisible final batch

    unbatched_inputs = torch.cat([inputs for _, inputs in unbatched_calls])
    batched_inputs = torch.cat([inputs for _, inputs in batched_calls])
    assert torch.equal(unbatched_inputs, batched_inputs)
    assert np.array_equal(unbatched_output, batched_output)


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5])
def test_global_wsi_rejects_invalid_batch_size(tmp_path, batch_size):
    image = np.ones((2, 16, 16), dtype=np.uint16)
    image_path = tmp_path / "invalid_batch_size.ome.tif"
    _write_test_ome(image_path, image)
    inst = object.__new__(InstanSeg)

    with pytest.raises(ValueError, match="positive integer"):
        inst.eval_whole_slide_image_global_normalization(
            image_path,
            channel_ids=[0],
            batch_size=batch_size,
            output_path=tmp_path / f"prediction_{str(batch_size)}.zarr",
        )


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
