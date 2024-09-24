#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 15:35:29 2024

@author: ian
"""
import torch
import os
from InstanSeg.utils.utils import download_model, _choose_device
from InstanSeg.utils.augmentations import Augmentations
import numpy as np
from skimage import io
from aicsimageio import AICSImage
import warnings
import torchvision
from torchvision.transforms import InterpolationMode
import math


class InstanSegModel:
    def __init__(self, model_folder, gpu=None):
        self.model_folder = model_folder
        if (gpu == True) or (gpu == None):
            self.device = torch.device(_choose_device(None))

        else:
            self.device = torch.device("cpu")

        print(f"### Device set: {self.device} ###")

        known_models = [
            "fluorescence_nuclei_and_cells",
            "brightfield_nuclei",
        ]
        self.model_path = os.path.join(model_folder, "instanseg.pt")

        if self.model_folder in known_models:
            if not os.path.isdir(self.model_folder):
                self.instanseg = download_model(
                    model_folder, return_model=True
                )
            else:
                self.instanseg = torch.jit.load(self.model_path)
            self.output_dimension = 2 if self.instanseg.cells_and_nuclei else 1
        else:
            if not os.path.isfile(self.model_path):
                print("There is no instanseg file in the model folder")
                return None
            self.instanseg = torch.jit.load(self.model_path)

        self.instanseg.to(self.device)

    def _process_labels(self, labels, original_shape):
        labels = torchvision.transforms.Resize(
            original_shape, interpolation=InterpolationMode.NEAREST
        )(labels)

        cpu_labels = labels.cpu().detach().numpy()
        if np.max(cpu_labels) <= 65535:
            cpu_labels = cpu_labels.astype(np.uint16)
        else:
            cpu_labels = cpu_labels.astype(np.uint32)
        cpu_labels_2D = []
        if len(cpu_labels.shape) == 4:
            for i in range(cpu_labels.shape[1]):
                slice_2D = cpu_labels[0, i, :, :]
                cpu_labels_2D.append(slice_2D)
                print(
                    f"Number of unique labels = {len(np.unique(slice_2D))-1}"
                )
        else:
            cpu_labels_2D = cpu_labels
        return cpu_labels_2D

    def _predict_small_image(self, input_tensor):
        if self.pixel_size is not None and not math.isnan(
            self.instanseg.pixel_size
        ):
            print(
                f"Rescaling image from {self.pixel_size} to match the model's pixel {self.instanseg.pixel_size} size"
            )
            input_tensor, _ = self.augmenter.torch_rescale(
                input_tensor,
                labels=None,
                current_pixel_size=self.pixel_size,
                requested_pixel_size=self.instanseg.pixel_size,
                crop=False,
            )

        if self.num_pixels > 3 * 1500 * 1500:
            from InstanSeg.utils.tiling import sliding_window_inference

            print("Processing using sliding window inference")

            labels = sliding_window_inference(
                input_tensor,
                self.instanseg,
                window_size=(self.tile_size, self.tile_size),
                overlap=50,
                max_cell_size=20,
                sw_device=self.device,
                device="cpu",
                output_channels=self.output_dimension,
                resolve_cell_and_nucleus=False,
            )

        else:
            print("Processing without tiling")
            with torch.amp.autocast("cuda"):
                labels = self.instanseg(input_tensor[None])
        return labels

    def _predict_large_image(self, image_path):
        from InstanSeg.utils.tiling import segment_image_larger_than_memory

        segment_image_larger_than_memory(
            instanseg_folder=self.model_folder,
            image_path=image_path,
            shape=(self.tile_size, self.tile_size),
            threshold=230,
            cell_size=50,
            overlap=50,
            to_geojson=self.output_geojson,
            driver="AUTO",
            torchscript=self.instanseg,
            pixel_size=self.pixel_size,
        )

    def _predict_from_filepath(self, image_path):
        image_name = os.path.basename(image_path)
        print("Processing file: ", image_name)
        img = AICSImage(image_path)
        if (self.pixel_size is None) and img.physical_pixel_sizes.X is None:
            warnings.warn(
                "Pixel size was not found in image metadata, please set pixel size of the input image in microns manually"
            )
        elif self.pixel_size is not None:
            pass
        elif img.physical_pixel_sizes.X is not None:
            self.pixel_size = img.physical_pixel_sizes.X
        else:
            return "Unexpected result"

        channel_number = img.dims.C
        self.num_pixels = np.cumprod(img.shape)[-1]
        print(f"Number of image pixels: {self.num_pixels}")
        if "S" in img.dims.order and img.dims.S > img.dims.C:
            channel_number = img.dims.S
            input_data = img.get_image_data("SYX")
        else:
            input_data = img.get_image_data("CYX")
        if self.process_with_zarr == False:
            input_tensor = self.augmenter.to_tensor(
                input_data, normalize=self.normalize
            )[0].to(self.device)
            original_shape = input_tensor.shape[1:]

            labels = self._predict_small_image(input_tensor)

        else:
            self._predict_large_image(input_data)

        labels = self._process_labels(labels, original_shape)
        return labels

    def _predict_from_array(self, image):
        print("Processing image")
        if len(image.shape) > 3:
            print(
                "Array input not implemented, try inputting image directly via filepath"
            )
            return None

        if self.pixel_size is not None:
            pass
        elif image.physical_pixel_sizes.X is not None:
            self.pixel_size = image.physical_pixel_sizes.X
        else:
            print("You need to specify a pixel size")
        self.num_pixels = np.cumprod(image.shape)[-1]

        input_tensor = self.augmenter.to_tensor(
            image, normalize=self.normalize
        )[0].to(self.device)
        original_shape = input_tensor.shape[1:]
        if self.process_with_zarr == False:
            labels = self._predict_small_image(input_tensor)
        else:
            print(
                "Zarr array input not implemented, try inputting image directly via filepath"
            )
        labels = self._process_labels(labels, original_shape)
        return labels

    def predict(
        self,
        input_data,
        pixel_size=None,
        normalize=True,
        tile_size=512,
        output_geojson=True,
        process_with_zarr=False,
    ):
        """
        Select
        """
        self.augmenter = Augmentations()
        self.pixel_size = pixel_size
        self.normalize = normalize
        self.tile_size = tile_size
        self.output_geojson = output_geojson
        self.process_with_zarr = process_with_zarr
        if isinstance(input_data, str):
            print("### Predicting from filepath using AICSimage ###")
            return self._predict_from_filepath(image_path=input_data)
        elif isinstance(input_data, np.ndarray):
            print("### Predicting from image array ###")
            return self._predict_from_array(image=input_data)


if __name__ == "__main__":
    # some testing code
    im = io.imread("brightfield_nuclei/sample_input_0.tif")
    IS_model = InstanSegModel(model_folder="brightfield_nuclei", gpu=False)
    labels_cpu = IS_model.predict(
        "brightfield_nuclei/sample_input_0.tif", pixel_size=0.4
    )
    labels_cpu_from_array = IS_model.predict(im, pixel_size=0.4)
    IS_model = InstanSegModel(model_folder="brightfield_nuclei", gpu=True)
    labels_gpu = IS_model.predict(
        "brightfield_nuclei/sample_input_0.tif", pixel_size=0.4
    )
    labels_gpu_from_array = IS_model.predict(im, pixel_size=0.4)
    io.imshow(labels_gpu_from_array[0])
