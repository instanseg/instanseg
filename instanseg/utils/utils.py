#!/usr/bin/python

import json
import os
import fastremap
import numpy as np
import tifffile
import torch
from matplotlib import pyplot as plt
import warnings
from typing import Union
from pathlib import Path
import rasterio.features
from rasterio.transform import Affine
from matplotlib.colors import LinearSegmentedColormap
import colorcet as cc
import matplotlib as mpl
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import tifffile
from typing import Optional, List
import geojson
## temporarily re-import display functions to avoid breaking code. Should be deprecated
from instanseg.utils.display import (
    moving_average,
    plot_average,
    apply_cmap,
    show_images,
    display_as_grid,
    generate_colors,
    color_name_to_rgb,
    save_image_with_label_overlay,
    display_cells_and_nuclei,
    display_colourized,
    _display_overlay,
    _to_rgb_channels_last,
    _to_scaled_uint8,
    _move_channel_axis
)

def timer(func: callable) -> callable:
    """
    A decorator that profiles the execution time of a function using LineProfiler.

    :param func: The function to be profiled.
    :return: The wrapped function with profiling.
    """
    import functools
    from line_profiler import LineProfiler
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        lp = LineProfiler()
        lp_wrapper = lp(func)
        lp_wrapper(*args, **kwargs)
        lp.print_stats()
        value = func(*args, **kwargs)
        return value

    return wrapper


def count_instances(labels: Union[np.ndarray, torch.Tensor]) -> int:
    """
    Count the total number of labelled pixels in an input tensor.
    :param labels: The input tensor.
    :return: The total number of non-zero labels.
    """
    import fastremap
    if isinstance(labels, torch.Tensor):
        num_labels = len(torch.unique(labels[labels > 0]))
    elif isinstance(labels, np.ndarray):
        num_labels = len(fastremap.unique(labels[labels > 0]))
    else:
        raise Exception("Labels must be numpy array or torch tensor")
    return num_labels



def set_export_paths():
    """
    Set the export paths for various model directories and ensure they exist.

    This function sets environment variables for the paths and creates the directories if they do not exist.
    """
    from pathlib import Path
    if os.environ.get('INSTANSEG_BIOIMAGEIO_PATH'):
        path = Path(os.environ['INSTANSEG_BIOIMAGEIO_PATH'])
    else:
        path = Path(os.path.join(os.path.dirname(__file__),"../bioimageio_models/"))
        os.environ['INSTANSEG_BIOIMAGEIO_PATH'] = str(path)

    if not path.exists():
        path.mkdir(exist_ok=True,parents=True)

    if os.environ.get('INSTANSEG_TORCHSCRIPT_PATH'):
        path = Path(os.environ['INSTANSEG_TORCHSCRIPT_PATH'])
    else:
        path = Path(os.path.join(os.path.dirname(__file__),"../torchscripts/"))
        os.environ['INSTANSEG_TORCHSCRIPT_PATH'] = str(path)

    if not path.exists():
        path.mkdir(exist_ok=True,parents=True)

    if os.environ.get('INSTANSEG_MODEL_PATH'):
        path = Path(os.environ['INSTANSEG_MODEL_PATH'])
    else:
        path = Path(os.path.join(os.path.dirname(__file__),"../models/"))
        os.environ['INSTANSEG_MODEL_PATH'] = str(path)

    if not path.exists():
        path.mkdir(exist_ok=True,parents=True)

    if os.environ.get('EXAMPLE_IMAGE_PATH'):
        path = Path(os.environ['EXAMPLE_IMAGE_PATH'])
    else:
        path = Path(os.path.join(os.path.dirname(__file__),"../examples/"))
        os.environ['EXAMPLE_IMAGE_PATH'] = str(path)


def export_to_torchscript(model_str: str,
                          show_example: bool = False,
                          output_dir: str = "../torchscripts",
                          model_path: str = "../models",
                          torchscript_name: str = None,
                          use_optimized_params = False):
    """
    Export a model to TorchScript format.

    :param model_str: The model string to export.
    :param show_example: Whether to show an example.
    :param output_dir: The directory to save the TorchScript model.
    :param model_path: The path to the model.
    :param torchscript_name: The name of the TorchScript model.
    :param use_optimized_params: Whether to use optimized parameters.
    """
    device = 'cpu'
    from instanseg.utils.model_loader import load_model
    import math

    import os
    set_export_paths()
    output_dir = os.environ.get('INSTANSEG_TORCHSCRIPT_PATH')
    model_path = os.environ.get('INSTANSEG_MODEL_PATH')
    example_path = os.environ.get('EXAMPLE_IMAGE_PATH')

    if use_optimized_params:
        import pandas as pd
        #Check is best_params.csv exists in the model folder, if not, use default parameters
        if not os.path.exists(Path(model_path) / model_str / "Results/best_params.csv"):
            print("No best_params.csv found in model folder, using default parameters")
            params = None
        else:
            #Load best_params.csv
            df = pd.read_csv(Path(model_path) / model_str / "Results/best_params.csv",header = None)
            params = {key: value for key, value in df.to_dict('tight')['data']}
    else:
        params = None

    model, model_dict = load_model(folder=model_str, path=model_path)
    model.eval()
    model.to(device)

    cells_and_nuclei = model_dict['cells_and_nuclei']
    pixel_size = model_dict['pixel_size']
    n_sigma = model_dict['n_sigma']


    input_data = tifffile.imread(os.path.join(example_path,"HE_example.tif"))
    #input_data = tifffile.imread("../examples/LuCa1.tif")
    from instanseg.utils.augmentations import Augmentations
    Augmenter = Augmentations()
    input_tensor, _ = Augmenter.to_tensor(input_data, normalize=False)
    input_tensor, _ = Augmenter.normalize(input_tensor)

    if not math.isnan(pixel_size):
        input_tensor, _ = Augmenter.torch_rescale(input_tensor, current_pixel_size=0.5, requested_pixel_size=pixel_size, crop =True, modality="Brightfield")
    input_tensor = input_tensor.to(device)


    if input_tensor.shape[0] != model_dict["dim_in"] and model_dict["dim_in"] != 0 and model_dict["dim_in"] is not None:
        input_tensor = torch.randn((model_dict["dim_in"], input_tensor.shape[1], input_tensor.shape[2])).to(device)
        dim_in = model_dict["dim_in"]
    else:
        dim_in = 3

    from instanseg.utils.loss.instanseg_loss import InstanSeg_Torchscript
    super_model = InstanSeg_Torchscript(model, cells_and_nuclei=cells_and_nuclei, 
                                        pixel_size = pixel_size, 
                                        n_sigma = n_sigma, 
                                        params = params, 
                                        feature_engineering_function = str(model_dict["feature_engineering"]), 
                                        backbone_dim_in= dim_in, 
                                        to_centre = bool(model_dict["to_centre"])).to(device)
    

    out = super_model(input_tensor[None,])
    if show_example:
        show_images([input_tensor] + [i for i in out.squeeze(0)], labels=[i + 1 for i in range(len(out.squeeze(0)))])

    with torch.jit.optimized_execution(should_optimize=True):
        traced_cpu = torch.jit.script(super_model, input_tensor[None,])

    if torchscript_name is None:
        torchscript_name = model_str

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    torch.jit.save(traced_cpu, os.path.join(output_dir, torchscript_name + ".pt"))
    print("Saved torchscript model to", os.path.join(output_dir, torchscript_name + ".pt"))


def export_annotations_and_images(output_dir: str, 
                                  original_image: np.ndarray, 
                                  lab: np.ndarray, 
                                  base_name: Optional[str] = None) -> None:
    """
    Export annotations and images to the specified directory.

    :param output_dir: The directory to save the annotations and images.
    :param original_image: The original image array.
    :param lab: The label array.
    :param base_name: The base name for the saved files.
    """
    from pathlib import Path
    import os
    if os.path.isfile(output_dir):
        base_name = Path(output_dir).stem
        output_dir = Path(output_dir).parent
    else:
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
            if base_name is None:
                base_name = "image"

    image = np.atleast_3d(np.squeeze(np.array(original_image)))
    image = _move_channel_axis(image)

    if image.shape[0] == 3:

        features = labels_to_features(lab.astype(np.int32), object_type='annotation', include_labels=True,
                                      classification=None)
        geojson = json.dumps(features)
        with open(os.path.join(output_dir, str(base_name) + '_labels.geojson'), "w") as outfile:
            outfile.write(geojson)

        save_image_with_label_overlay(image, lab, output_dir=output_dir, label_boundary_mode='thick', alpha=0.8,
                                      base_name=base_name)

    else:
        warnings.warn("Did not attempt to save image of shape:")
        print(image.shape)

def drag_and_drop_file() -> str:
    """
    Open a window where a user can drop a file and return the path to the file.

    :return: The path to the dropped file.
    """
    import tkinter as tk
    from tkinterdnd2 import TkinterDnD, DND_FILES

    def drop(event):
        file_path = event.data
        entry_var.set(file_path)

    def save_and_close():
        entry_var.get()
        root.destroy()  # Close the window

    root = TkinterDnD.Tk()
    entry_var = tk.StringVar()
    entry = tk.Entry(root, textvariable=entry_var, width=40)
    entry.pack(pady=20)

    entry.drop_target_register(DND_FILES)
    entry.dnd_bind('<<Drop>>', drop)

    save_button = tk.Button(root, text="Save and Close", command=save_and_close)
    save_button.pack(pady=10)
    root.mainloop()
    return entry_var.get()



def labels_to_features(lab: np.ndarray,
                       object_type='annotation',
                       connectivity: int = 4,
                       transform: Affine = None,
                       downsample: float = 1.0,
                       include_labels: bool = False,
                       classification: str = None,
                       offset: int = None) -> geojson.FeatureCollection:
    """
    Create a GeoJSON FeatureCollection from a labeled image.

    :param classification: Optional classification added to output features.
    :param offset: Offset added to x coordinates of output features.
    :return: A GeoJSON FeatureCollection.
    """
    features = []

    # Ensure types are valid
    if lab.dtype == bool:
        mask = lab
        lab = lab.astype(np.uint8)
    else:
        mask = lab != 0

    # Create transform from downsample if needed
    if transform is None:
        transform = Affine.scale(downsample)

    # Trace geometries
    for i, obj in enumerate(rasterio.features.shapes(lab, mask=mask,
                                                     connectivity=connectivity, transform=transform)):

        # Create properties
        props = dict(object_type=object_type)
        if include_labels:
            props['measurements'] = [{'name': 'Label', 'value': i}]
        #  pdb.set_trace()

        # Just to show how a classification can be added
        if classification is not None:
            props['classification'] = classification

        if offset is not None:
            coordinates = obj[0]['coordinates']
            coordinates = [
                [(int(x[0] + offset[0]), int(x[1] + offset[1])) for x in coordinates[0]]]
            obj[0]['coordinates'] = coordinates

        po = geojson.Feature(geometry = obj[0], properties=props)

        features.append(po)
    return geojson.FeatureCollection(features)



def percentile_normalize(img: Union[np.ndarray, torch.Tensor],
                         percentile: float = 0.1,
                         subsampling_factor: int = 1,
                         epsilon: float = 1e-3) -> Union[np.ndarray, torch.Tensor]:
    """
    Normalize an image using percentile normalization.

    :param img: The input image array or tensor.
    :param percentile: The percentile for normalization.
    :param subsampling_factor: The subsampling factor.
    :param epsilon: A small value to avoid division by zero.
    :return: The normalized image array or tensor.
    """

    if isinstance(img, np.ndarray):
        assert img.ndim == 2 or img.ndim == 3, "Image must be 2D or 3D, got image of shape" + str(img.shape)
        img = np.atleast_3d(img)
        channel_axis = np.argmin(img.shape)
        img = _move_channel_axis(img, to_back=True)

        for c in range(img.shape[-1]):
            im_temp = img[::subsampling_factor, ::subsampling_factor, c]
            (p_min, p_max) = np.percentile(im_temp, [percentile, 100 - percentile])
            img[:, :, c] = (img[:, :, c] - p_min) / max(epsilon, p_max - p_min)
       # img = img / np.maximum(0.01, np.max(img))
        return np.moveaxis(img, 2, channel_axis)

    elif isinstance(img, torch.Tensor):
        assert img.ndim == 2 or img.ndim == 3, "Image must be 2D or 3D, got image of shape" + str(img.shape)
        img = torch.atleast_3d(img)
        channel_axis = np.argmin(img.shape)
        img = _move_channel_axis(img, to_back=True)
        for c in range(img.shape[-1]):
            im_temp = img[::subsampling_factor, ::subsampling_factor, c]
            if img.is_cuda or img.is_mps:
                (p_min, p_max) = torch.quantile(im_temp, torch.tensor([percentile / 100, (100 - percentile) / 100],device = im_temp.device))
            else:
                (p_min, p_max) = np.percentile(im_temp.cpu(), [percentile, 100 - percentile])
            img[:, :, c] = (img[:, :, c] - p_min) / max(epsilon, p_max - p_min)
       # img = img / np.maximum(0.01, torch.max(img))
        return img.movedim(2, channel_axis)



def _choose_device(device: str = None, verbose=True) -> str:
    """
    Choose a device to use with PyTorch, given the desired device name.
    If a requested device is not specified or not available, then a default is chosen.
    """
    if device is not None:
        if device == 'cuda' and not torch.cuda.is_available():
            device = None
            print('CUDA device requested but not available!')
        if device == 'mps' and not torch.backends.mps.is_available():
            device = None
            print('MPS device requested but not available!')

    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
        if verbose:
            print(f'Requesting default device: {device}')

    return device



def _estimate_image_modality(img, mask):
    """
    This function estimates the modality of an image (i.e. brightfield, chromogenic or fluorescence) based on the
    mean intensity of the pixels inside and outside the mask.
    """

    if isinstance(img, np.ndarray):
        img = np.atleast_3d(_move_channel_axis(img))
        mask = np.squeeze(mask)

        assert mask.ndim == 2, print("Mask must be 2D, but got shape", mask.shape)
        if count_instances(mask) < 1:
            return "Fluorescence"  # The images don't contain any cells, but they look like fluorescence images.
        elif np.mean(img[:, mask > 0]) > np.mean(img[:, mask == 0]):
            return "Fluorescence"
        else:
            if img.shape[0] == 1:
                return "Brightfield"
            else:
                return "Brightfield"  # "Chromogenic"


    elif isinstance(img, torch.Tensor):
        img = torch.at_least_3d(_move_channel_axis(img))
        mask = torch.squeeze(mask)
        assert mask.ndim == 2, "Mask must be 2D"

        if count_instances(mask) < 1:
            return "Fluorescence"  # The images don't contain any cells, but they look like fluorescence images.
        elif torch.mean(img[:, mask > 0]) > torch.mean(img[:, mask == 0]):
            return "Fluorescence"
        else:
            if img.shape[0] == 1:
                return "Brightfield"
            else:
                return "Brightfield"  # "Chromogenic"
