# Quickstart

This page shows how to segment an image with a pretrained InstanSeg model. It assumes you have
already {doc}`installed <installation>` InstanSeg.

:::{note}
The examples use `HE_example.tif`, an H&E image from the
[`instanseg/examples`](https://github.com/instanseg/instanseg/tree/main/instanseg/examples)
folder of the repository. Replace it with the path to your own image.
:::

## Segment an image

Create an {class}`~instanseg.InstanSeg` object with the name of a pretrained model, then call
{meth}`~instanseg.InstanSeg.eval` on the path to an image:

```python
from instanseg import InstanSeg

instanseg_brightfield = InstanSeg("brightfield_nuclei", image_reader="tiffslide", verbosity=1)

labeled_output = instanseg_brightfield.eval(
    image="HE_example.tif",
    save_output=True,
    save_overlay=True,
)
```

The model is downloaded the first time it is used. `eval` returns the labels as a
{class}`torch.Tensor`, where each object has its own integer ID and the background is `0`. With
`save_output` and `save_overlay` set, the labels and an overlay of the labels on the image are
also saved next to the input image.

## Work with arrays

For more control over each step, read the image yourself and pass the array to
{meth}`~instanseg.InstanSeg.eval_small_image`:

```python
from instanseg.utils.utils import show_images

image_array, pixel_size = instanseg_brightfield.read_image("HE_example.tif")

labeled_output, image_tensor = instanseg_brightfield.eval_small_image(image_array, pixel_size)

display = instanseg_brightfield.display(image_tensor, labeled_output)

show_images(image_tensor, display, colorbar=False, titles=["Normalized image", "Image with segmentation"])
```

`eval_small_image` returns the labels and the normalised input image, which
{meth}`~instanseg.InstanSeg.display` combines into an RGB overlay.

## Next steps

- {doc}`inference` covers the pretrained models, pixel sizes, large images and the command line.
- {doc}`training` explains how to train InstanSeg on your own data.
- The {doc}`api/index` documents every public class and function.
