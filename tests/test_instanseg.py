from pathlib import Path
from instanseg.instanseg import InstanSeg
import torch
import sys

def test_inference():
    import os
    sys.path = sys.path[1:]

    example_image_folder = Path(os.path.join(os.path.dirname(__file__),"../instanseg/examples/"))
    print(example_image_folder)

    instanseg_brightfield = InstanSeg("brightfield_nuclei", verbosity=0)
    image_array, pixel_size = instanseg_brightfield.read_image(example_image_folder/"HE_example.tif")
    labeled_output, image_tensor  = instanseg_brightfield.eval_small_image(image_array, pixel_size)
    display = instanseg_brightfield.display(image_tensor, labeled_output)

    instanseg_fluorescence = InstanSeg("fluorescence_nuclei_and_cells", verbosity=0)
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

