#!/usr/bin/python
"""
Visualization utilities for InstanSeg.

This module contains functions for displaying images, creating overlays,
and plotting training metrics.
"""

import os
from typing import List, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
from skimage.color import label2rgb


def _moving_average(x, w):
    """Moving average of an array x with window size w"""
    return np.convolve(x, np.ones(w), 'valid') / w


def plot_average(train, test, clip=99, window_size=10):
    """Plot smoothed training and test loss curves."""
    fig = plt.figure(figsize=(10, 10))
    clip_val = np.percentile(test, [clip])
    test = np.clip(test, 0, clip_val[0])
    clip_val = np.percentile(train, [clip])
    train = np.clip(train, 0, clip_val[0])
    plt.plot(_moving_average(test, window_size), label="test")
    plt.plot(_moving_average(train, window_size), label="train")
    plt.legend()
    return fig


def apply_cmap(x,
               fg_mask: np.ndarray = None,
               cmap: str = "coolwarm_r",
               bg_intensity: int = 255,
               normalize: bool = True):
    """
    Apply a colormap to an image, with a background mask.
    x and fg_mask should have the same shape, and be numpy arrays.
    """
    import matplotlib.cm as cm
    import matplotlib
    norm = matplotlib.colors.Normalize(vmin=1, vmax=10)

    if fg_mask is None:
        fg_mask = (x != 0).squeeze()
    x_clone = x.copy().astype(np.float32)
    x_clone = x_clone.squeeze()
    fg_clone = x_clone[fg_mask]
    cmap = matplotlib.colormaps.get_cmap(cmap)
    m = cm.ScalarMappable(cmap=cmap, norm=norm)

    rgba_image = m.to_rgba(fg_clone)
    canvas = np.zeros((x_clone.shape[0], x_clone.shape[1], 4)).astype(np.float32)
    canvas[fg_mask] = rgba_image
    canvas = (canvas[:, :, :3] * 255).astype(np.uint8)
    canvas[fg_mask == 0] = bg_intensity
    return canvas


def label_to_color_image(label_img, **kwargs):
    """
    Convert a labelled image to a color image using the Glasbey colormap.
    Label 0 -> black, label -1 -> white, other labels -> Glasbey colors.
    
    Parameters:
        label_img (np.ndarray): 2D array of shape (H, W) with integer labels.
        
    Returns:
        np.ndarray: RGB image of shape (H, W, 3), dtype=np.uint8
    """
    if label_img.ndim != 2:
        raise ValueError("Input image must be a 2D array.")

    # Ensure it's int type
    label_img = label_img.astype(int)

    # Find unique labels excluding -1 and 0
    labels = np.unique(label_img)
    labels = labels[(labels != 0) & (labels != -1)]
    
    # Create a dummy image with only the positive labels to use with label2rgb
    label_img_tmp = np.copy(label_img)
    label_img_tmp[label_img_tmp <= 0] = 0

    # Apply Glasbey colormap using label2rgb
    color_img = label2rgb(label_img_tmp, bg_label=0, **kwargs)
    color_img = (color_img * 255).astype(np.uint8)

    # Map 0 to black
    color_img[label_img == 0] = [0, 0, 0]

    # Map -1 to white
    color_img[label_img == -1] = [255, 255, 255]

    return color_img


def show_images(*img_list, clip_pct=None, titles=None, save_str=False, n_cols=3, axes=False, cmap="viridis",
                labels=None,
                dpi=None, timer_flag=None, colorbar=True, **args):
    """Designed to plot torch tensor and numpy arrays in windows robustly"""

    if labels is None:
        labels = []
    if titles is None:
        titles = []
    if dpi:
        mpl.rcParams['figure.dpi'] = dpi

    img_list = [img for img in img_list]
    if isinstance(img_list[0], list):
        img_list = img_list[0]

    rows = (len(img_list) - 1) // n_cols + 1
    columns = np.min([n_cols, len(img_list)])
    fig = plt.figure(figsize=(5 * (columns + 1), 5 * (rows + 1)))

    fig.tight_layout()
    grid = plt.GridSpec(rows, columns, figure=fig)
    grid.update(wspace=0.2, hspace=0, left=None, right=None, bottom=None, top=None)

    for i, img in enumerate(img_list):

        if torch.is_tensor(img):
            img = torch.squeeze(img).detach().cpu().numpy()
        img = np.squeeze(img)
        if len(img.shape) > 2:
            img = np.moveaxis(img, np.argmin(img.shape), -1)
            if img.shape[-1] > 4 or img.shape[-1] == 2:
                plt.close()
                show_images([img[..., i] for i in range(img.shape[-1])], clip_pct=clip_pct,
                            titles=["Channel:" + str(i) for i in range(img.shape[-1])], save_str=save_str,
                            n_cols=n_cols, axes=axes, cmap=cmap, colorbar=colorbar)
                continue
        ax1 = plt.subplot(grid[i])
        if not axes:
            plt.axis('off')
        if clip_pct is not None:
            print(np.percentile(img.ravel(), clip_pct), np.percentile(img.ravel(), 100 - clip_pct))
            im = ax1.imshow(img, vmin=np.percentile(img.ravel(), clip_pct),
                            vmax=np.percentile(img.ravel(), 100 - clip_pct))
        if i in labels:
            try:
                import colorcet as cc
                colors = cc.cm.glasbey_bw_minc_20_minl_30_r.colors
                img = label_to_color_image(img, colors=colors)
            except ImportError:
                img = label_to_color_image(img)
            im = ax1.imshow(img, interpolation='nearest')
        else:
            im = ax1.imshow(img, cmap=cmap, **args)
            if colorbar and img.shape[-1] != 3:
                plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
        if i < len(titles):
            ax1.set_title(titles[i])
    if not save_str:
        if timer_flag is not None:
            plt.show(block=False)
            plt.pause(timer_flag)
            plt.close()

        plt.show()

    if save_str:
        plt.savefig(str(save_str) + ".png", bbox_inches='tight')
        plt.close()
        return None


def display_as_grid(display_list,
                    ncols: int,
                    padding: int = 2,
                    left_titles: Optional[List[str]] = None,
                    top_titles=None,
                    right_side=None,
                    title_height: int = 20,
                    fontsize: float = 12):
    """Display a list of images in a grid layout."""
    from instanseg.utils.augmentations import Augmentations
    Augmenter = Augmentations()

    tensor_list = []
    for i in display_list:
        disp_tensor = Augmenter.to_tensor(i, normalize=False)[0].to("cpu")
        h, w = disp_tensor.shape[1:]
        tensor_list.append(disp_tensor / disp_tensor.max())

    from torchvision.utils import make_grid

    grid = make_grid(tensor_list, nrow=ncols, padding=padding, pad_value=1)

    fig = plt.figure(figsize=(10, 10))
    plt.imshow(grid.numpy().transpose(1, 2, 0))
    plt.axis('off')

    if left_titles is not None:
        for idx, dataset in enumerate(left_titles):
            plt.text(-title_height, idx * h + int((h / 2)) + 2 * idx, dataset, fontsize=fontsize, color='black',
                     verticalalignment='center', rotation="vertical")
    if top_titles is not None:
        for idx, dataset in enumerate(top_titles):
            plt.text(idx * w + int((w / 2)), -title_height, dataset, fontsize=fontsize, color='black',
                     verticalalignment='center', horizontalalignment='center', rotation="horizontal")
    if right_side is not None:
        for idx, dataset in enumerate(right_side):
            plt.text(5 * w + 10, idx * h + int((h / 2)) + 2 * idx, dataset, fontsize=fontsize, color='black',
                     verticalalignment='center', rotation=270)

    return fig


def _color_name_to_rgb(color_name: str):
    """
    Convert a color name to its corresponding RGB values.
    
    :param color_name: The name of the color.
    :return: A tuple containing the RGB values.
    """
    return mcolors.to_rgb(color_name)


def _to_scaled_uint8(im: np.ndarray, clip_percentile=1.0) -> np.ndarray:
    """
    Convert an image to uint8, scaling according to the given percentile.
    """
    im_float = im.astype(np.float32, copy=True)
    min_val = np.percentile(im_float.ravel(), clip_percentile)
    max_val = np.percentile(im_float.ravel(), 100.0 - clip_percentile)
    im_float -= min_val
    im_float /= (max_val - min_val)
    im_float *= 255
    return np.clip(im_float, a_min=0, a_max=255).astype(np.uint8)


def _to_rgb_channels_last(im: np.ndarray,
                          clip_percentile: float = 1.0,
                          scale_per_channel: bool = True,
                          input_channels_first: bool = True) -> np.ndarray:
    """
    Convert an image to RGB, ensuring the output has channels-last ordering.
    """
    if im.ndim < 2 or im.ndim > 3:
        raise ValueError(f"Number of dimensions should be 2 or 3! Image has shape {im.shape}")
    if im.ndim == 3:
        if input_channels_first:
            im = np.moveaxis(im, source=0, destination=-1)
        if im.shape[-1] != 3:
            im = im.mean(axis=-1)
    if im.ndim > 2 and scale_per_channel:
        im_scaled = np.dstack(
            [_to_scaled_uint8(im[..., ii], clip_percentile=clip_percentile) for ii in range(3)]
        )
    else:
        im_scaled = _to_scaled_uint8(im, clip_percentile=clip_percentile)
    if im.ndim == 2:
        im_scaled = np.repeat(im_scaled, repeats=3, axis=-1)
    return im_scaled


def _move_channel_axis_viz(img, to_back: bool = False):
    """Move channel axis for visualization (internal helper)."""
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


def save_image_with_label_overlay(im: np.ndarray,
                                  lab: np.ndarray,
                                  output_dir: str = "./",
                                  base_name: str = 'image',
                                  clip_percentile: float = 1.0,
                                  scale_per_channel: bool = None,
                                  label_boundary_mode="inner",
                                  label_colors=None,
                                  alpha=1.0,
                                  thickness=3,
                                  return_image=False):
    """
    Save an image as RGB alongside a corresponding label overlay.
    This can be used to quickly visualize the results of a segmentation, generally using the
    default image viewer of the operating system.

    :param im: Image to save. If this is an 8-bit RGB image (channels-last) it will be used as-is,
               otherwise it will be converted to RGB
    :param lab: Label image to overlay
    :param output_dir: Directory to save to
    :param base_name: Base name for the image files to save
    :param clip_percentile: Percentile to clip the image at. Used during RGB conversion, if needed.
    :param scale_per_channel: Whether to scale each channel independently during RGB conversion.
    :param label_boundary_mode: A boundary mode compatible with scikit-image find_boundaries;
                                one of 'thick', 'inner', 'outer', 'subpixel'. If None, the lab is used directly.
    :param alpha: Alpha value for the underlying image when using the overlay.
                  Setting this less than 1 can help the overlay stand out more prominently.
    :return: The input image with corresponding label overlay if lab is a torch Tensor, or nothing otherwise.
    """
    import imageio
    from skimage.segmentation import find_boundaries
    from skimage import morphology

    if isinstance(im, torch.Tensor):
        im = torch.clamp(im, 0, 1).cpu().numpy() * 255
        im = _move_channel_axis_viz(im, to_back=True).astype(np.uint8)

    if isinstance(lab, torch.Tensor):
        lab = _move_channel_axis_viz(torch.atleast_3d(lab.squeeze())).cpu().numpy()

        if lab.shape[0] == 1:
            lab = lab[0]
            image_overlay = save_image_with_label_overlay(im, lab=lab, return_image=True, label_boundary_mode="thick",
                                                          label_colors=label_colors, thickness=5, alpha=1)
        elif lab.shape[0] == 2:
            nuclei_labels_for_display = lab[0]
            cell_labels_for_display = lab[1]
            bg = (lab.sum(0) == 0)

            if label_boundary_mode is None:
                try:
                    from palettable.scientific import diverging as div
                    colour_cells = list((np.array(div.Berlin_12.colors[1]) / 255))
                    colour_nuclei = list(np.array(div.Berlin_12.colors[11]) / 255)
                except ImportError:
                    colour_cells = [0.2, 0.6, 0.8]
                    colour_nuclei = [0.8, 0.2, 0.2]
                image_overlay = save_image_with_label_overlay(im, lab=cell_labels_for_display, return_image=True,
                                                              label_boundary_mode=None, label_colors=colour_cells,
                                                              thickness=1, alpha=1)
                image_overlay = save_image_with_label_overlay(image_overlay, lab=nuclei_labels_for_display,
                                                              return_image=True, label_boundary_mode=None,
                                                              label_colors=colour_nuclei, thickness=1, alpha=1)
                image_overlay = save_image_with_label_overlay(image_overlay, lab=nuclei_labels_for_display,
                                                              return_image=True, label_boundary_mode="thick",
                                                              label_colors="black", thickness=1, alpha=1)
                image_overlay = save_image_with_label_overlay(image_overlay, lab=cell_labels_for_display,
                                                              return_image=True, label_boundary_mode="thick",
                                                              label_colors="black", thickness=1, alpha=1)
                image_overlay[bg] = (255, 255, 255)
            else:
                image_overlay = save_image_with_label_overlay(im, lab=cell_labels_for_display, return_image=True,
                                                              label_boundary_mode=label_boundary_mode,
                                                              label_colors="cyan", thickness=1, alpha=1)
                image_overlay = save_image_with_label_overlay(image_overlay, lab=nuclei_labels_for_display,
                                                              return_image=True,
                                                              label_boundary_mode=label_boundary_mode,
                                                              label_colors="magenta", thickness=1, alpha=1)
        return image_overlay

    # Check if we have an RGB, channels-last image already
    if im.dtype == np.uint8 and im.ndim == 3 and im.shape[2] == 3:
        im_rgb = im.copy()
    else:
        im_rgb = _to_rgb_channels_last(im, clip_percentile=clip_percentile, scale_per_channel=scale_per_channel)

    # Convert labels to boundaries, if required
    if label_boundary_mode is not None:
        bw_boundaries = find_boundaries(lab, mode=label_boundary_mode)
        lab = lab.copy()
        # Need to expand labels for outer boundaries, but have to avoid overwriting known labels
        if label_boundary_mode in ['thick', 'outer']:
            lab2 = morphology.dilation(lab, footprint=np.ones((thickness, thickness)))
            mask_dilated = bw_boundaries & (lab == 0)

            lab[mask_dilated] = lab2[mask_dilated]
        lab[~bw_boundaries] = 0
        mask_temp = bw_boundaries
    else:
        mask_temp = lab != 0

    # Create a labels displayed
    if label_colors is None:
        lab_overlay = label2rgb(lab).astype(np.float32)
    elif label_colors == "red":
        lab_overlay = np.zeros((lab.shape[0], lab.shape[1], 3))
        lab_overlay[lab > 0, 0] = 1
    elif label_colors == "green":
        lab_overlay = np.zeros((lab.shape[0], lab.shape[1], 3))
        lab_overlay[lab > 0, 1] = 1
    elif label_colors == "blue":
        lab_overlay = np.zeros((lab.shape[0], lab.shape[1], 3))
        lab_overlay[lab > 0, 2] = 1
    else:
        if type(label_colors) == str:
            label_colors = _color_name_to_rgb(label_colors)
        lab_overlay = np.zeros((lab.shape[0], lab.shape[1], 3))
        lab_overlay[lab > 0] = label_colors

    im_overlay = im_rgb.copy()
    for c in range(im_overlay.shape[-1]):
        im_temp = im_overlay[..., c]
        lab_temp = lab_overlay[..., c]
        im_temp[mask_temp] = (lab_temp[mask_temp] * 255).astype(np.uint8)

    # Make main image translucent, if needed
    if alpha != 1.0:
        alpha_channel = (mask_temp * 255.0).astype(np.uint8)
        alpha_channel[~mask_temp] = alpha * 255
        im_overlay = np.dstack((im_overlay, alpha_channel))
    if return_image:
        return im_overlay

    if base_name is None:
        base_name = "image"

    print(f"Exporting image to {os.path.join(output_dir, f'{base_name}.png')}")
    imageio.imwrite(os.path.join(output_dir, f'{base_name}.png'), im_rgb)
    imageio.imwrite(os.path.join(output_dir, f'{base_name}_overlay.png'), im_overlay)


def display_cells_and_nuclei(lab):
    """Display cells and nuclei labels as a colored overlay."""
    display = save_image_with_label_overlay(torch.zeros((lab.shape[-2], lab.shape[-1], 3)), lab, return_image=True,
                                            label_boundary_mode=None, alpha=1)
    return display


def display_colourized(mIF, random_seed=0):
    """Convert a multi-channel image to a colorized RGB display."""
    from instanseg.utils.augmentations import Augmentations
    Augmenter = Augmentations()

    mIF = Augmenter.to_tensor(mIF, normalize=False)[0]
    if mIF.shape[0] != 3:
        colour_render, _ = Augmenter.colourize(mIF, random_seed=random_seed)
    else:
        colour_render = Augmenter.to_tensor(mIF, normalize=True)[0]
    colour_render = torch.clamp_(colour_render, 0, 1)
    colour_render = _move_channel_axis_viz(colour_render, to_back=True).detach().numpy() * 255
    return colour_render.astype(np.uint8)


def _display_overlay(im, lab):
    """Create an overlay of labels on an image for display."""
    assert lab.ndim == 4, "lab must be 4D"
    assert im.ndim == 3, "im must be 3D"
    output_dimension = lab.shape[1]

    im_for_display = display_colourized(im)

    if output_dimension == 1:  # Nucleus or cell mask
        labels_for_display = lab[0, 0].cpu().numpy()  # Shape is 1,H,W
        image_overlay = save_image_with_label_overlay(im_for_display, lab=labels_for_display, return_image=True,
                                                      label_boundary_mode="thick", label_colors=None, thickness=10,
                                                      alpha=0.5)
    elif output_dimension == 2:  # Nucleus and cell mask
        nuclei_labels_for_display = lab[0, 0].cpu().numpy()
        cell_labels_for_display = lab[0, 1].cpu().numpy()  # Shape is 1,H,W
        image_overlay = save_image_with_label_overlay(im_for_display, lab=nuclei_labels_for_display, return_image=True,
                                                      label_boundary_mode="thick", label_colors="red", thickness=10)
        image_overlay = save_image_with_label_overlay(image_overlay, lab=cell_labels_for_display, return_image=True,
                                                      label_boundary_mode="inner", label_colors="green", thickness=1)
    return image_overlay

