#!/usr/bin/python
"""
Core utilities for InstanSeg.

This module contains essential helper functions for image processing,
model loading, and device management.
"""

import json
import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

import fastremap
import numpy as np
import tifffile
import torch

if TYPE_CHECKING:
    import geojson


# Re-export visualization functions for backwards compatibility
from instanseg.utils.visualization import (
    apply_cmap,
    display_as_grid,
    display_cells_and_nuclei,
    display_colourized,
    label_to_color_image,
    plot_average,
    save_image_with_label_overlay,
    show_images,
    _display_overlay,
)


def labels_to_features(lab: np.ndarray,
                       object_type='annotation',
                       connectivity: int = 4,
                       transform: Any = None,
                       downsample: float = 1.0,
                       include_labels: bool = False,
                       classification: str = None,
                       offset: int = None) -> "geojson.FeatureCollection":
    """
    Create a GeoJSON FeatureCollection from a labeled image.

    :param classification: Optional classification added to output features.
    :param offset: Offset added to x coordinates of output features.
    :return: A GeoJSON FeatureCollection.
    
    Note: Requires 'rasterio' and 'geojson' packages. Install with: pip install instanseg-torch[io]
    """
    try:
        import geojson
        import rasterio.features
        from rasterio.transform import Affine
    except ImportError as e:
        raise ImportError(
            "GeoJSON export requires 'rasterio' and 'geojson' packages. "
            "Install with: pip install instanseg-torch[io]"
        ) from e

    features = []

    # Ensure types are valid
    if lab.dtype == bool:
        mask = lab
        lab = lab.astype(np.uint8)
    else:
        mask = lab != 0

    # Create transform from downsample if needed
    if transform is None:
        transform = Affine.scale(1.0)

    # Trace geometries
    for i, obj in enumerate(rasterio.features.shapes(lab, mask=mask,
                                                     connectivity=connectivity, transform=transform)):

        # Create properties
        props = dict(object_type=object_type)
        if include_labels:
            props['measurements'] = [{'name': 'Label', 'value': i}]

        # Just to show how a classification can be added
        if classification is not None:
            props['classification'] = classification

        if offset is not None:
            coordinates = obj[0]['coordinates']
            coordinates = [
                [(round(x[0] * downsample) + offset[0], round(x[1] * downsample) + offset[1]) for x in coordinates[0]]]
            obj[0]['coordinates'] = coordinates

        po = geojson.Feature(geometry=obj[0], properties=props)

        features.append(po)
    return geojson.FeatureCollection(features)


def _move_channel_axis(img: Union[np.ndarray, torch.Tensor], to_back: bool = False) -> Union[np.ndarray, torch.Tensor]:
    """Move the channel axis to the front or back of an image tensor."""
    if isinstance(img, np.ndarray):
        img = img.squeeze()
        if img.ndim != 3:
            if img.ndim == 2:
                img = img[None, ]
            if img.ndim != 3:
                raise ValueError("Input array should be 3D or 2D")
        ch = np.argmin(img.shape)
        if to_back:
            return np.rollaxis(img, ch, 3)

        return np.rollaxis(img, ch, 0)
    elif isinstance(img, torch.Tensor):
        if img.dim() != 3:
            if img.dim() == 2:
                img = img[None, ]
            if img.dim() != 3:
                raise ValueError("Input array should be 3D or 2D")
        ch = np.argmin(img.shape)
        if to_back:
            return img.movedim(ch, -1)
        return img.movedim(ch, 0)


def percentile_normalize(img: Union[np.ndarray, torch.Tensor],
                         percentile: float = 0.1,
                         subsampling_factor: int = 1,
                         epsilon: float = 1e-3):
    """Normalize an image using percentile-based scaling."""
    if isinstance(img, np.ndarray):
        assert img.ndim == 2 or img.ndim == 3, "Image must be 2D or 3D, got image of shape" + str(img.shape)
        img = np.atleast_3d(img)
        channel_axis = np.argmin(img.shape)
        img = _move_channel_axis(img, to_back=True)

        for c in range(img.shape[-1]):
            im_temp = img[::subsampling_factor, ::subsampling_factor, c]
            (p_min, p_max) = np.percentile(im_temp, [percentile, 100 - percentile])
            img[:, :, c] = (img[:, :, c] - p_min) / max(epsilon, p_max - p_min)
        return np.moveaxis(img, 2, channel_axis)

    elif isinstance(img, torch.Tensor):
        assert img.ndim == 2 or img.ndim == 3, "Image must be 2D or 3D, got image of shape" + str(img.shape)
        img = torch.atleast_3d(img)
        channel_axis = np.argmin(img.shape)
        img = _move_channel_axis(img, to_back=True)
        for c in range(img.shape[-1]):
            im_temp = img[::subsampling_factor, ::subsampling_factor, c]
            if img.is_cuda or img.is_mps:
                (p_min, p_max) = torch.quantile(im_temp,
                                                torch.tensor([percentile / 100, (100 - percentile) / 100],
                                                             device=im_temp.device))
            else:
                (p_min, p_max) = np.percentile(im_temp.cpu(), [percentile, 100 - percentile])
            img[:, :, c] = (img[:, :, c] - p_min) / max(epsilon, p_max - p_min)
        return img.movedim(2, channel_axis)


def generate_colors(num_colors: int):
    """Generate evenly spaced colors in HSV space."""
    import colorsys
    # Calculate the equally spaced hue values
    hues = [i / float(num_colors) for i in range(num_colors)]

    # Generate RGB colors
    rgb_colors = [list(colorsys.hsv_to_rgb(hue, 1.0, 1.0)) for hue in hues]

    return rgb_colors


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


def count_instances(labels: Union[np.ndarray, torch.Tensor]) -> int:
    """
    Count the total number of labelled pixels in an input tensor.
    :param labels: The input tensor.
    :return: The total number of non-zero labels.
    """
    if isinstance(labels, torch.Tensor):
        num_labels = len(torch.unique(labels[labels > 0]))
    elif isinstance(labels, np.ndarray):
        num_labels = len(fastremap.unique(labels[labels > 0]))
    else:
        raise Exception("Labels must be numpy array or torch tensor")
    return num_labels


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


def timer(func):
    """Decorator for profiling function execution time using line_profiler."""
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


def set_export_paths():
    """Set environment variables for model export paths."""
    if os.environ.get('INSTANSEG_BIOIMAGEIO_PATH'):
        path = Path(os.environ['INSTANSEG_BIOIMAGEIO_PATH'])
    else:
        path = Path(os.path.join(os.path.dirname(__file__), "../bioimageio_models/"))
        os.environ['INSTANSEG_BIOIMAGEIO_PATH'] = str(path)

    if not path.exists():
        path.mkdir(exist_ok=True, parents=True)

    if os.environ.get('INSTANSEG_TORCHSCRIPT_PATH'):
        path = Path(os.environ['INSTANSEG_TORCHSCRIPT_PATH'])
    else:
        path = Path(os.path.join(os.path.dirname(__file__), "../torchscripts/"))
        os.environ['INSTANSEG_TORCHSCRIPT_PATH'] = str(path)

    if not path.exists():
        path.mkdir(exist_ok=True, parents=True)

    if os.environ.get('INSTANSEG_MODEL_PATH'):
        path = Path(os.environ['INSTANSEG_MODEL_PATH'])
    else:
        path = Path(os.path.join(os.path.dirname(__file__), "../models/"))
        os.environ['INSTANSEG_MODEL_PATH'] = str(path)

    if not path.exists():
        path.mkdir(exist_ok=True, parents=True)

    if os.environ.get('EXAMPLE_IMAGE_PATH'):
        path = Path(os.environ['EXAMPLE_IMAGE_PATH'])
    else:
        path = Path(os.path.join(os.path.dirname(__file__), "../examples/"))
        os.environ['EXAMPLE_IMAGE_PATH'] = str(path)


def export_to_torchscript(model_str: str, show_example: bool = False, output_dir: str = "../torchscripts",
                          model_path: str = "../models", torchscript_name: str = None, use_optimized_params=False):
    """Export an InstanSeg model to TorchScript format."""
    device = 'cpu'
    from instanseg.utils.model_loader import load_model
    import math

    set_export_paths()
    output_dir = os.environ.get('INSTANSEG_TORCHSCRIPT_PATH')
    model_path = os.environ.get('INSTANSEG_MODEL_PATH')
    example_path = os.environ.get('EXAMPLE_IMAGE_PATH')

    if use_optimized_params:
        import pandas as pd
        # Check is best_params.csv exists in the model folder, if not, use default parameters
        if not os.path.exists(Path(model_path) / model_str / "Results/best_params.csv"):
            print("No best_params.csv found in model folder, using default parameters")
            params = None
        else:
            # Load best_params.csv
            df = pd.read_csv(Path(model_path) / model_str / "Results/best_params.csv", header=None)
            params = {key: value for key, value in df.to_dict('tight')['data']}
    else:
        params = None

    model, model_dict = load_model(folder=model_str, path=model_path)
    model.eval()
    model.to(device)

    cells_and_nuclei = model_dict['cells_and_nuclei']
    pixel_size = model_dict['pixel_size']
    n_sigma = model_dict['n_sigma']

    input_data = tifffile.imread(os.path.join(example_path, "HE_example.tif"))
    from instanseg.utils.augmentations import Augmentations
    Augmenter = Augmentations()
    input_tensor, _ = Augmenter.to_tensor(input_data, normalize=False)
    input_tensor, _ = Augmenter.normalize(input_tensor)

    if not math.isnan(pixel_size):
        input_tensor, _ = Augmenter.torch_rescale(input_tensor, current_pixel_size=0.5, requested_pixel_size=pixel_size,
                                                  crop=True, modality="Brightfield")
    input_tensor = input_tensor.to(device)

    if input_tensor.shape[0] != model_dict["dim_in"] and model_dict["dim_in"] != 0 and model_dict["dim_in"] is not None:
        input_tensor = torch.randn((model_dict["dim_in"], input_tensor.shape[1], input_tensor.shape[2])).to(device)
        dim_in = model_dict["dim_in"]
    else:
        dim_in = 3

    from instanseg.utils.loss.instanseg_loss import InstanSeg_Torchscript
    super_model = InstanSeg_Torchscript(model, cells_and_nuclei=cells_and_nuclei,
                                        pixel_size=pixel_size,
                                        n_sigma=n_sigma,
                                        params=params,
                                        feature_engineering_function=str(model_dict["feature_engineering"]),
                                        backbone_dim_in=dim_in).to(device)

    out = super_model(input_tensor[None, ])
    if show_example:
        show_images([input_tensor] + [i for i in out.squeeze(0)], labels=[i + 1 for i in range(len(out.squeeze(0)))])

    with torch.jit.optimized_execution(should_optimize=True):
        traced_cpu = torch.jit.script(super_model, input_tensor[None, ])

    if torchscript_name is None:
        torchscript_name = model_str

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    torch.jit.save(traced_cpu, os.path.join(output_dir, torchscript_name + ".pt"))
    print("Saved torchscript model to", os.path.join(output_dir, torchscript_name + ".pt"))


def drag_and_drop_file():
    """
    This opens a window where a user can drop a file and returns the path to the file.
    
    Note: Requires 'tkinterdnd2' package. Install with: pip install instanseg-torch[full]
    """
    try:
        import tkinter as tk
        from tkinterdnd2 import TkinterDnD, DND_FILES
    except ImportError as e:
        raise ImportError(
            "drag_and_drop_file requires 'tkinterdnd2' package. "
            "Install with: pip install instanseg-torch[full]"
        ) from e

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


def download_model(model_str: str, version: Optional[str] = None, verbose: bool = True, force: bool = False):
    """Download and load an InstanSeg model from the model repository."""
    import requests
    import zipfile
    from io import BytesIO
    from pkgutil import get_data

    if not os.environ.get("INSTANSEG_BIOIMAGEIO_PATH"):
        os.environ["INSTANSEG_BIOIMAGEIO_PATH"] = os.path.join(os.path.dirname(__file__), "../bioimageio_models/")

    bioimageio_path = os.environ.get("INSTANSEG_BIOIMAGEIO_PATH")
    os.makedirs(bioimageio_path, exist_ok=True)

    output = get_data("instanseg", "bioimageio_models/model-index.json")
    content = output.decode('utf-8')
    models = json.loads(content)

    model = [model for model in models if model["name"] == model_str]
    if version is not None and len(model):
        model = [model for model in models if model["version"] == version]

    if len(model):
        model = model[0]  # if we're not specifying version, then pick the first (newest)
        url = model["url"]
        output_path = Path(bioimageio_path) / model["name"] / model["version"]
        path_to_torchscript_model = output_path / "instanseg.pt"

        if os.path.isdir(output_path) and os.path.exists(path_to_torchscript_model) and not force:
            if verbose:
                print(f"Model {model['name']} version {model['version']} already downloaded in {bioimageio_path}, loading")
            return torch.jit.load(path_to_torchscript_model)

        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses
        with zipfile.ZipFile(BytesIO(response.content)) as z:
            os.makedirs(output_path, exist_ok=True)
            z.extractall(output_path)

        if verbose:
            print(f"Model {model['name']} version {model['version']} downloaded and extracted to {bioimageio_path}")

        return torch.jit.load(path_to_torchscript_model)

    else:
        # load model locally
        model_path = model_str
        if version is not None:
            if verbose:
                print(f"Assuming model is stored under {bioimageio_path}/{model_str}/{version}...")
            model_path = model_str + os.path.sep + version
        path_to_torchscript_model = os.path.join(bioimageio_path, model_path, "instanseg.pt")

        if os.path.exists(path_to_torchscript_model):
            return torch.jit.load(path_to_torchscript_model)
        else:
            raise Exception(
                f"Model {path_to_torchscript_model} version {version} not found in the release data or locally. Please check the model name and try again.")


def _filter_kwargs(func, kwargs):
    """Filter kwargs to only include those accepted by a TorchScript function."""
    graph_str = str(func.graph).split("):\n")[0]
    lines = graph_str.split("\n")
    arg_lines = [line.strip() for line in lines if "%" in line and ":" in line]
    arg_names = [line.split(":")[0].strip().replace("%", "").replace(".1", "") for line in arg_lines]
    arg_names = [name for name in arg_names if name not in ['graph(self', 'x', 'args']]
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in arg_names}

    return filtered_kwargs
