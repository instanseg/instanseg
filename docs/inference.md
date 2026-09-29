# Running inference

## Pretrained models

Pass one of these names to {class}`~instanseg.InstanSeg` to download and use a pretrained model:

| Model | Images | Outputs |
| --- | --- | --- |
| `brightfield_nuclei` | Brightfield (e.g. H&E) | Nuclei |
| `fluorescence_nuclei_and_cells` | Fluorescence and multiplexed, any number of channels | Nuclei and cells |
| `single_channel_nuclei` | Single-channel fluorescence only | Nuclei |

Models are downloaded once and cached. You can also pass a {class}`torch.nn.Module` to use a
model you have trained or loaded yourself.

For models that predict both nuclei and cells, the labels have two channels: nuclei first, then
cells. Use the `target` argument of {meth}`~instanseg.InstanSeg.eval_small_image` and
{meth}`~instanseg.InstanSeg.eval_medium_image` to get only `"nuclei"` or `"cells"`.

## Pixel size

InstanSeg rescales images to the pixel size its model was trained on, so it needs to know the
pixel size of your image, in microns. It is read from the image metadata when possible. If the
metadata is missing, or you want to override it, pass `pixel_size`:

```python
labels = instanseg.eval("image.tif", pixel_size=0.5)
```

If no pixel size is given or found, InstanSeg warns you and segments the image at its original
scale, which may give inaccurate results.

## Image size and processing methods

{meth}`~instanseg.InstanSeg.eval` picks a processing method based on the size of the image. To
choose one yourself, set `processing_method`:

`"small"`
: The whole image is segmented in one pass, on the GPU if available. Used for images up to about
  1500 × 1500 pixels with 3 channels. See {meth}`~instanseg.InstanSeg.eval_small_image`.

`"medium"`
: The image is loaded into memory, split into tiles, and the objects are merged across tiles.
  Used for images up to about 10,000 × 10,000 pixels. See
  {meth}`~instanseg.InstanSeg.eval_medium_image`.

`"wsi"`
: For whole slide images too large to load into memory. The slide is read and segmented tile by
  tile, and the labels are written to a `.zarr` file next to the image. See
  {meth}`~instanseg.InstanSeg.eval_whole_slide_image`.

:::{note}
Support for whole slide images is limited. Reading them requires the `io` extra:
`pip install "instanseg-torch[io]"`.
:::

## Saving results

`eval` can save its results next to each input image, with `_instanseg_prediction` added to the
file name:

`save_output=True`
: The labels, as a `.tiff` file.

`save_overlay=True`
: The labels drawn over the image.

`save_geojson=True`
: The object outlines as a GeoJSON feature collection, which can be opened in QuPath. Requires
  the `io` extra.

You can also save results yourself with {meth}`~instanseg.InstanSeg.save_output`.

## Several images at once

Pass a list of paths to `eval` to segment each image in turn. It returns a list of labels, in the
same order:

```python
labels = instanseg.eval(["image_1.tif", "image_2.tif"], save_output=True)
```

## Command line

Installing InstanSeg adds an `inference` command that segments every image in a folder and saves
the labels and overlays next to each image:

```console
$ inference --model_folder brightfield_nuclei --image_path path/to/images
```

`--model_folder` accepts a pretrained model name or a model you have trained. Useful options:

| Option | Description |
| --- | --- |
| `--pixel_size` | Pixel size in microns, if it can't be read from the image metadata. |
| `--recursive True` | Also look for images in subfolders. |
| `--ignore_segmented True` | Skip images that already have a prediction. |
| `--save_geojson True` | Also save GeoJSON outlines. |
| `--device` | Device to run on, e.g. `cpu` or `cuda:0`. |
| `--image_reader` | Image reader to use: `auto`, `tiffslide`, `skimage.io`, `bioio` or `AICSImageIO`. |
| `--tile_size`, `--batch_size` | Tiling settings for large images. |

Any other `key=value` pairs passed with `--kwargs` are forwarded to
{meth}`~instanseg.InstanSeg.eval`.
