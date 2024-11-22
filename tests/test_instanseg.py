import torch
import sys
import os
import numpy as np

from pathlib import Path
from instanseg import InstanSeg


def test_inference():
    sys.path = sys.path[1:]

    example_image_folder = Path(os.path.join(os.path.dirname(__file__),"../instanseg/examples/"))
    print(example_image_folder)

    device = "cuda" if sys.platform == "linux" else "cpu"
    instanseg_brightfield = InstanSeg("brightfield_nuclei", verbosity=0, device=device)
    image_array, pixel_size = instanseg_brightfield.read_image(example_image_folder/"HE_example.tif")
    labeled_output, image_tensor  = instanseg_brightfield.eval_small_image(image_array, pixel_size)
    display = instanseg_brightfield.display(image_tensor, labeled_output)

    instanseg_fluorescence = InstanSeg("fluorescence_nuclei_and_cells", verbosity=0, device=device)
    image_array, pixel_size = instanseg_fluorescence.read_image(example_image_folder/"Fluorescence_example.tif")

    labeled_output, image_tensor  = instanseg_fluorescence.eval_small_image(image_array, pixel_size)
    display = instanseg_fluorescence.display(image_tensor, labeled_output)

    instanseg_fluorescence.eval_small_image(torch.randn(1,1,256,256), 0.5)
    
    instanseg_fluorescence.eval_small_image(torch.randn(1,1,256,256), 1, rescale_output = True)
    
    instanseg_fluorescence.eval_small_image(torch.randn(1,1,256,256), 1, rescale_output=False)
    
    instanseg_fluorescence.eval_small_image(torch.randn(1,1,256,256), 1,
                                             rescale_output=False, 
                                             return_image_tensor=False,
                                             normalise = False)
    
    input = torch.randn(1,1,1,1,1,1,1000,1000)
    label,img = instanseg_fluorescence.eval_medium_image(input, 0.5, rescale_output=True)
    assert label.shape[-2:] == input.shape[-2:]

    input = torch.randn(1,1,1000,256)
    label,img = instanseg_fluorescence.eval_medium_image(input, 0.5, rescale_output=True)
    assert label.shape[-2:] == input.shape[-2:]

    input = torch.randn(1,5,512,512)
    label,img = instanseg_fluorescence.eval_medium_image(input, 0.5, rescale_output=True)
    assert label.shape[-2:] == input.shape[-2:]

    input = torch.randn(1,1,256,256)
    label,img = instanseg_fluorescence.eval_medium_image(input, 1, rescale_output=True, normalise = True)
    assert label.shape[-2:] == input.shape[-2:]

    input = torch.randn(1,1,2560,2560)
    label,img = instanseg_fluorescence.eval_medium_image(input, 0.1, rescale_output=True)
    assert label.shape[-2:] == input.shape[-2:]


def test_image_readers():
    sys.path = sys.path[1:]

    example_image_folder = Path(os.path.join(os.path.dirname(__file__),"../instanseg/examples/"))
    print(example_image_folder)
    fluoro_outputs=[]
    he_outputs=[]
    for reader in ["bioio", "skimage.io", "tiffslide"]:

        device = "cuda" if sys.platform == "linux" else "cpu"
        instanseg_brightfield = InstanSeg("brightfield_nuclei", verbosity=0, device=device, image_reader=reader)
        labeled_output = instanseg_brightfield.eval(str(example_image_folder/"HE_example.tif")).cpu().numpy()

        he_outputs.append(labeled_output)
        instanseg_fluorescence = InstanSeg("fluorescence_nuclei_and_cells", verbosity=0, device=device)
        image_array, pixel_size = instanseg_fluorescence.read_image(str(example_image_folder/"Fluorescence_example.tif"))

        labeled_output = [x.cpu().numpy() for x in instanseg_fluorescence.eval_small_image(image_array)]
        fluoro_outputs.append(labeled_output)

    np.testing.assert_equal(he_outputs[0], he_outputs[1])
    np.testing.assert_equal(he_outputs[1], he_outputs[2])
    np.testing.assert_equal(fluoro_outputs[0][0], fluoro_outputs[1][0])
    np.testing.assert_equal(fluoro_outputs[0][1], fluoro_outputs[1][1])
    np.testing.assert_equal(fluoro_outputs[1][0], fluoro_outputs[2][0])
    np.testing.assert_equal(fluoro_outputs[1][1], fluoro_outputs[2][1])



def test_whole_slide_image():
    sys.path = sys.path[1:]

    example_image_folder = Path(os.path.join(os.path.dirname(__file__),"../instanseg/examples/"))
    print(example_image_folder)
    for reader in ["bioio", "skimage.io", "tiffslide"]:

        device = "cuda" if sys.platform == "linux" else "cpu"
        instanseg_brightfield = InstanSeg("brightfield_nuclei", verbosity=0, device=device, image_reader=reader, vram_size_threshold=10, ram_size_threshold=10)
        instanseg_brightfield.eval_whole_slide_image(str(example_image_folder/"HE_example.tif"))

        he_outputs.append(labeled_output)
        instanseg_fluorescence = InstanSeg("fluorescence_nuclei_and_cells", verbosity=0, device=device, vram_size_threshold=10, ram_size_threshold=10)

        instanseg_fluorescence.eval_whole_slide_image(image_array)

